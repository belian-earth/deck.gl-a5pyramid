// Reading and packing tiles, in a worker (or under Node in tests): no DOM,
// no deck.gl. Ported from a5view's tileCore (lazy-renderer.js) for shards of
// a set: a shard's footer arrives as bytes fetched once by the main thread,
// and its tiles are read with range requests from the shard's URL.

import { parquetMetadata, parquetRead, type Compressors, type FileMetaData } from "hyparquet";
import { cellToBoundary, cellToLonLat } from "a5-js";
import { readOffsetIndex } from "hyparquet/src/indexes.js";
import { readColumn } from "hyparquet/src/column.js";
import { getTreePath } from "hyparquet/src/schema.js";
import { DEFAULT_PARSERS } from "hyparquet/src/convert.js";
import { parquetSchema, type ColumnMetaData } from "hyparquet";
import { cellSize } from "./tiling.js";
import { chunkRange, listColumn, readFixedList, type ListColumn } from "./fixed-list.js";

/** A tile ready for deck.gl: binary polygon attributes plus what hover needs. */
export interface PackedTile {
  n: number;
  lod: number;
  cells: BigUint64Array;
  ncell: Uint32Array;
  /** One per scalar value column, the stored codes. */
  values: Int8Array[];
  /** The vector column (an embedding), `dim` codes a cell, row by row; null without one. */
  vector: Int8Array | null;
  dim: number;
  /**
   * Every code of the tile laid out as an RGBA8 texture for the GPU colouring
   * (codesLayout): `codesRows` rows of `codesWidth` texels, each cell's codes
   * in `texels` consecutive texels of one row, `perRow` cells a row. The
   * vector, when its width is a multiple of 4, is a view of these bytes.
   */
  codes: Uint8Array;
  codesWidth: number;
  codesRows: number;
  texels: number;
  perRow: number;
  /** Vertex start of each cell, n + 1 entries. */
  starts: Uint32Array;
  /** lon, lat per vertex (open rings). */
  positions: Float64Array;
  /** Fan triangles over each ring, absolute vertex indices; null to let deck triangulate. */
  indices: Uint32Array | null;
  /** Bytes the tile holds in memory, colours included; deck.gl's tile cache reads it. */
  byteLength: number;
  /** Bytes fetched to read the tile, and timings (ms): reading (fetch and
   * hyparquet's columns), decoding the vector and filling the codes, and
   * packing the boundaries. A piece that shared its row group's read
   * carries none of the read. */
  fetched: number;
  readMs: number;
  decodeMs: number;
  packMs: number;
  /** The tile's sums for a PCA of its decoded values (when asked for). */
  stats: TileStats | null;
  statsMs: number;
}

/**
 * What a PCA of the values needs from a tile, summed so tiles combine by
 * adding: the total weight, the weighted sum of each column and the
 * weighted sum of each product of two columns (upper triangle, row by row).
 * Values are decoded (AEF codes to embedding values) and weighted by ncell,
 * so a coarse cell stands for its leaves.
 */
export interface TileStats {
  d: number;
  w: number;
  sum: Float64Array;
  cross: Float64Array;
}

/** AEF codes as the embedding values they stand for: sign(x) * (x / 127.5)^2. */
export const AEF_DECODE = (() => {
  const t = new Float64Array(256);
  for (let c = -128; c < 128; c++) {
    const x = c / 127.5;
    t[c + 128] = Math.sign(c) * x * x;
  }
  return t;
})();

/** Rows a tile's PCA sums are taken from at most: an even sample of a larger
 * tile, scaled to its whole weight, keeps the cost per tile near fixed. 1,024
 * of a 16,384-row tile: a view still samples 15 to 90 k cells for 64
 * dimensions, and the sums cost a quarter of 4,096's (measured 2026-11-12:
 * 0.9 to 0.2 s of worker time over an eight-view tour). */
export const STATS_ROWS = 1024;
/** The tile size STATS_ROWS is set for: a larger tile or row group samples
 * no more sparsely than one of these. */
export const STATS_TILE = 16384;

/** A tile's PCA sums over its vector column, or over its scalar columns.
 * Rows are sampled every `every` rows (default: STATS_ROWS of the tile). */
export function tileStats(t: { values: Int8Array[]; vector: Int8Array | null; dim: number; ncell: Uint32Array; n: number }, every?: number): TileStats {
  const { values, vector, ncell, n } = t;
  const d = vector ? t.dim : values.length;
  const sum = new Float64Array(d);
  const cross = new Float64Array((d * (d + 1)) / 2);
  const x = new Float64Array(d);
  const step = Math.max(1, every ?? Math.ceil(Math.min(n, STATS_TILE) / STATS_ROWS));
  let wAll = 0;
  for (let i = 0; i < n; i++) wAll += ncell[i];
  let wSample = 0;
  for (let i = 0; i < n; i += step) {
    const w = ncell[i];
    wSample += w;
    if (vector) for (let k = 0; k < d; k++) x[k] = AEF_DECODE[vector[i * d + k] + 128];
    else for (let k = 0; k < d; k++) x[k] = AEF_DECODE[values[k][i] + 128];
    let at = 0;
    for (let j = 0; j < d; j++) {
      const wx = w * x[j];
      sum[j] += wx;
      for (let k = j; k < d; k++) cross[at++] += wx * x[k];
    }
  }
  const scale = wSample > 0 ? wAll / wSample : 0;
  for (let j = 0; j < d; j++) sum[j] *= scale;
  for (let j = 0; j < cross.length; j++) cross[j] *= scale;
  return { d, w: wAll, sum, cross };
}

/** Texels a row of a codes texture at most (WebGL 2 guarantees 2048). */
export const CODES_WIDTH = 2048;
/** Codes a cell the GPU colouring reads at most: 16 RGBA texels. */
export const MAX_INPUTS = 64;

/** Where a tile of `n` cells with `d` codes each puts them in its texture. */
export function codesLayout(n: number, d: number) {
  const texels = Math.max(1, Math.ceil(Math.min(d, MAX_INPUTS) / 4));
  const perRow = Math.floor(CODES_WIDTH / texels);
  return { texels, perRow, width: perRow * texels, rows: Math.max(1, Math.ceil(n / perRow)) };
}

/** Cells at this LOD or coarser are triangulated by deck (earcut): their
 * segmented edges need not be convex in lon/lat. Finer ones get fans. */
export const FAN_MIN_LOD = 9;

interface Shard {
  metadata: FileMetaData;
  rowStarts: number[];
  file: RemoteFile;
  /** Each vector column's list layout, null for a fixed binary one. */
  lists: Map<string, ListColumn | null>;
  /** Whether the shard has an ncell column (leaves of store v4 do not). */
  hasNcell?: boolean;
  /** The schema tree, for decoding pages apart from parquetRead. */
  tree?: ReturnType<typeof parquetSchema>;
}

/** Decompressors beyond hyparquet's own (snappy): store schema v4 files are
 * zstd. A worker that reads them sets these before its first tile. */
let compressors: Compressors | undefined;
export function setCompressors(c: Compressors | undefined): void {
  compressors = c;
}

/** A shard file read by HTTP range requests. */
class RemoteFile {
  constructor(readonly url: string, readonly byteLength: number) {}
  async slice(start: number, end?: number): Promise<ArrayBuffer> {
    const last = (end ?? this.byteLength) - 1;
    const res = await fetch(this.url, { headers: { Range: `bytes=${start}-${last}` } });
    if (res.status !== 206 && res.status !== 200) throw new Error(`${this.url}: HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    // a server that ignores Range sends the whole file
    return res.status === 200 && buf.byteLength === this.byteLength ? buf.slice(start, last + 1) : buf;
  }
}

const shards = new Map<string, Shard>();

/** Parse a shard's footer (its last `footer_bytes` bytes) and keep it. */
export function openShard(url: string, byteLength: number, footer: ArrayBuffer): FileMetaData {
  const known = shards.get(url);
  if (known) return known.metadata;
  const metadata = parquetMetadata(footer);
  const rowStarts = [0];
  for (const rg of metadata.row_groups) rowStarts.push(rowStarts[rowStarts.length - 1] + Number(rg.num_rows));
  shards.set(url, { metadata, rowStarts, file: new RemoteFile(url, byteLength), lists: new Map() });
  return metadata;
}

export function hasShard(url: string): boolean {
  return shards.has(url);
}

/** The key-value metadata of a parsed footer. */
export function keyValues(metadata: FileMetaData): Record<string, string> {
  const out: Record<string, string> = {};
  for (const e of metadata.key_value_metadata ?? []) out[e.key] = e.value ?? "";
  return out;
}

/**
 * How a tile is read (the viewer's optimisations of 2026-11-12, measured one
 * by one): as a whole row group, or a piece of it of `pieceRows` rows.
 */
export interface ReadOptions {
  /** Rows a piece; with `piece`, the tile's rows [piece * pieceRows, +pieceRows) alone. */
  pieceRows?: number;
  piece?: number;
  /** Read the piece's pages alone through the page index (pages aligned to pieces). */
  pages?: boolean;
  /** The row group's columns in one range request, not one a column chunk
   * (default true: 57% fewer requests over a tour, no more bytes to speak
   * of, since a tile's chunks lie together). */
  singleRange?: boolean;
  /** PCA sums from every `statsEvery`-th row (default: STATS_ROWS a tile). */
  statsEvery?: number;
}

/** A row group's rows, decoded. */
interface Rows {
  n: number;
  cells: BigUint64Array;
  ncell: Uint32Array;
  values: Int8Array[];
  vector: Int8Array | null;
  fetched: number;
  readMs: number;
  decodeMs: number;
}

/** Row groups decoded in this worker, most recent last: pieces of one tile
 * share a read. */
const decoded = new Map<string, Promise<Rows>>();
const DECODED_KEEP = 24;

function cached<T>(m: Map<string, Promise<T>>, key: string, keep: number, make: () => Promise<T>): Promise<T> {
  const hit = m.get(key);
  if (hit) {
    m.delete(key);
    m.set(key, hit);
    return hit;
  }
  const p = make();
  m.set(key, p);
  p.catch(() => m.delete(key));
  while (m.size > keep) m.delete(m.keys().next().value!);
  return p;
}

/** A shard's file as hyparquet reads it, counting the bytes fetched. */
function counted(s: Shard, count: (n: number) => void) {
  return {
    byteLength: s.file.byteLength,
    slice: async (start: number, end?: number) => {
      const buf = await s.file.slice(start, end);
      count(buf.byteLength);
      return buf;
    },
  };
}

/** The list layout of the vector column of `s`, or null for fixed binary. */
function vectorList(s: Shard, vector: { name: string; dim: number } | null): ListColumn | null {
  if (!vector) return null;
  if (!s.lists.has(vector.name)) s.lists.set(vector.name, listColumn(s.metadata, vector.name));
  return s.lists.get(vector.name) ?? null;
}

/** Decode rows of the flat columns from hyparquet's chunks. */
function flatRows(n: number, chunks: Map<string, ArrayLike<unknown>[]>, columns: string[], hasNcell: boolean) {
  const cells = new BigUint64Array(n);
  copyInto(cells, chunks.get("cell") ?? [], (v) => (typeof v === "bigint" ? BigInt.asUintN(64, v) : BigInt(v as number)));
  const ncell = new Uint32Array(n);
  if (hasNcell) copyInto(ncell, chunks.get("ncell") ?? [], (v) => Number(v));
  else ncell.fill(1);
  const values = columns.map((name) => {
    const out = new Int8Array(n);
    copyInto(out, chunks.get(name) ?? [], (v) => (v == null ? 0 : Number(v)));
    return out;
  });
  return { cells, ncell, values };
}

/** Read a whole row group: every requested column, decoded. */
async function readRows(s: Shard, tile: number, columns: string[], vector: { name: string; dim: number } | null, opts: ReadOptions): Promise<Rows> {
  const t0 = performance.now();
  let fetched = 0;
  const list = vectorList(s, vector);
  if (s.hasNcell === undefined) s.hasNcell = s.metadata.schema.some((e) => e.name === "ncell");
  const rg = s.metadata.row_groups[tile];
  const wanted = new Set(["cell", ...(s.hasNcell ? ["ncell"] : []), ...columns, ...(vector ? [vector.name] : [])]);
  const n = s.rowStarts[tile + 1] - s.rowStarts[tile];
  // one range for every wanted column chunk of the group (their chunks lie
  // together), served to hyparquet and the list reader from memory
  let file = counted(s, (b) => (fetched += b));
  if (opts.singleRange ?? true) {
    let lo = Infinity, hi = 0;
    for (const c of rg.columns) {
      const m = c.meta_data!;
      if (!wanted.has(m.path_in_schema[0])) continue;
      const [a, b] = chunkRange(m);
      lo = Math.min(lo, a);
      hi = Math.max(hi, b);
    }
    const buf = await s.file.slice(lo, hi);
    fetched += buf.byteLength;
    file = { byteLength: s.file.byteLength, slice: async (a: number, b?: number) => buf.slice(a - lo, (b ?? s.file.byteLength) - lo) };
  }
  const listMeta = list ? rg.columns[list.index].meta_data! : null;
  const chunks = new Map<string, ArrayLike<unknown>[]>();
  const [, listBytes] = await Promise.all([
    parquetRead({
      file, metadata: s.metadata, compressors,
      rowStart: s.rowStarts[tile], rowEnd: s.rowStarts[tile + 1],
      columns: [...wanted].filter((c) => !(list && c === vector!.name)),
      onChunk: (c) => {
        const part = chunks.get(c.columnName) ?? [];
        part.push(c.columnData as ArrayLike<unknown>);
        chunks.set(c.columnName, part);
      },
    }),
    listMeta ? file.slice(...chunkRange(listMeta)) : Promise.resolve(null),
  ]);
  const t1 = performance.now();
  const { cells, ncell, values } = flatRows(n, chunks, columns, !!s.hasNcell);
  let vec: Int8Array | null = null;
  if (vector) {
    vec = new Int8Array(n * vector.dim);
    if (list && listMeta && listBytes) readFixedList(new Uint8Array(listBytes), listMeta, list, vector.dim, vec, compressors);
    else {
      let i = 0;
      for (const part of chunks.get(vector.name) ?? []) {
        for (let j = 0; j < part.length && i < n; j++, i++) {
          const row = part[j] as Uint8Array;
          vec.set(new Int8Array(row.buffer, row.byteOffset, vector.dim), i * vector.dim);
        }
      }
      if (i !== n) throw new Error(`vector column has ${i} rows for ${n}`);
    }
  }
  return { n, cells, ncell, values, vector: vec, fetched, readMs: t1 - t0, decodeMs: performance.now() - t1 };
}

/** A row group's page locations (the offset index) and dictionary pages,
 * for the columns a piece reads, kept per tile. */
interface TilePages {
  cols: Map<string, { meta: ColumnMetaData; pages: { offset: number; size: number; first: number }[]; dict: Uint8Array }>;
  fetched: number;
}
const tilePages = new Map<string, Promise<TilePages>>();

/** The offset indexes of `names` in row group `tile` (one request: they lie
 * together at the end of the file) and their dictionary pages (one each). */
async function pagesOf(s: Shard, tile: number, names: string[]): Promise<TilePages> {
  const rg = s.metadata.row_groups[tile];
  const cols = rg.columns.filter((c) => names.includes(c.meta_data!.path_in_schema[0]));
  if (cols.some((c) => c.offset_index_offset == null)) throw new Error("no page index: pieces need one");
  const lo = Math.min(...cols.map((c) => Number(c.offset_index_offset)));
  const hi = Math.max(...cols.map((c) => Number(c.offset_index_offset) + Number(c.offset_index_length)));
  const buf = await s.file.slice(lo, hi);
  let fetched = buf.byteLength;
  const out: TilePages["cols"] = new Map();
  await Promise.all(cols.map(async (c) => {
    const meta = c.meta_data!;
    const at = Number(c.offset_index_offset) - lo;
    const pages = readOffsetIndex({ view: new DataView(buf, at, Number(c.offset_index_length)), offset: 0 }).page_locations.map((l) => ({
      offset: Number(l.offset), size: l.compressed_page_size, first: Number(l.first_row_index),
    }));
    const start = chunkRange(meta)[0];
    let dict = new Uint8Array(0);
    if (pages.length && start < pages[0].offset) {
      dict = new Uint8Array(await s.file.slice(start, pages[0].offset));
      fetched += dict.byteLength;
    }
    out.set(meta.path_in_schema[0], { meta, pages, dict });
  }));
  return { cols: out, fetched };
}

/** Read piece `piece` of a row group whose pages are cut at the piece's rows:
 * one request a column for its page (the tile's page index and dictionaries
 * read once a worker), flat columns decoded by hyparquet, the list vector by
 * the flat list reader. */
async function readPiecePages(s: Shard, url: string, tile: number, piece: number, pieceRows: number, columns: string[],
  vector: { name: string; dim: number } | null): Promise<Rows> {
  const t0 = performance.now();
  const list = vectorList(s, vector);
  if (s.hasNcell === undefined) s.hasNcell = s.metadata.schema.some((e) => e.name === "ncell");
  s.tree ??= parquetSchema(s.metadata);
  const tileRows = s.rowStarts[tile + 1] - s.rowStarts[tile];
  const r0 = piece * pieceRows;
  const n = Math.min(pieceRows, tileRows - r0);
  const flat = ["cell", ...(s.hasNcell ? ["ncell"] : []), ...columns, ...(vector && !list ? [vector.name] : [])];
  const names = [...flat, ...(list ? [vector!.name] : [])];
  let first = false;
  const tp = await cached(tilePages, `${url}#${tile}`, 64, () => ((first = true), pagesOf(s, tile, names)));
  let fetched = first ? tp.fetched : 0;
  // each column's page of the piece, all at once
  const bytes = new Map<string, Uint8Array>();
  await Promise.all(names.map(async (name) => {
    const c = tp.cols.get(name);
    if (!c) throw new Error(`no column ${name}`);
    const k = c.pages.findIndex((l) => l.first === r0);
    if (k < 0) throw new Error(`no page of ${name} starts at row ${r0} of tile ${tile}: pages are not cut at ${pieceRows} rows`);
    const end = k + 1 < c.pages.length ? c.pages[k + 1].first : tileRows;
    if (end - r0 !== n) throw new Error(`page ${k} of ${name} holds ${end - r0} rows, the piece ${n}`);
    const page = new Uint8Array(await s.file.slice(c.pages[k].offset, c.pages[k].offset + c.pages[k].size));
    fetched += page.byteLength;
    const both = new Uint8Array(c.dict.length + page.length);
    both.set(c.dict);
    both.set(page, c.dict.length);
    bytes.set(name, both);
  }));
  const t1 = performance.now();
  const chunks = new Map<string, ArrayLike<unknown>[]>();
  for (const name of flat) {
    const c = tp.cols.get(name)!;
    const b = bytes.get(name)!;
    const schemaPath = getTreePath(s.tree, c.meta.path_in_schema);
    const decoder = {
      pathInSchema: c.meta.path_in_schema, element: schemaPath[schemaPath.length - 1].element, schemaPath,
      type: c.meta.type, codec: c.meta.codec, compressors, parsers: DEFAULT_PARSERS, utf8: true,
    };
    const { data } = readColumn({ view: new DataView(b.buffer, b.byteOffset, b.byteLength), offset: 0 },
      { groupStart: 0, groupRows: n, selectStart: 0, selectEnd: n }, decoder as any);
    chunks.set(name, data as ArrayLike<unknown>[]);
  }
  const { cells, ncell, values } = flatRows(n, chunks, columns, !!s.hasNcell);
  let vec: Int8Array | null = null;
  if (vector) {
    vec = new Int8Array(n * vector.dim);
    if (list) {
      const c = tp.cols.get(vector.name)!;
      readFixedList(bytes.get(vector.name)!, { ...c.meta, num_values: BigInt(n * vector.dim) }, list, vector.dim, vec, compressors);
    } else {
      let i = 0;
      for (const part of chunks.get(vector.name) ?? []) {
        for (let j = 0; j < part.length && i < n; j++, i++) {
          const row = part[j] as Uint8Array;
          vec.set(new Int8Array(row.buffer, row.byteOffset, vector.dim), i * vector.dim);
        }
      }
    }
  }
  return { n, cells, ncell, values, vector: vec, fetched, readMs: t1 - t0, decodeMs: performance.now() - t1 };
}

/** Pack rows [a, b) of decoded rows into a tile. */
function packRows(r: Rows, a: number, b: number, lod: number, dimOrNull: number | null, withStats: boolean, statsEvery: number | undefined,
  timing: { fetched: number; readMs: number; decodeMs: number }): PackedTile {
  const n = b - a;
  const cells = r.cells.slice(a, b);
  const ncell = r.ncell.slice(a, b);
  const values = r.values.map((v) => v.slice(a, b));
  const dim = dimOrNull ?? 0;
  const layout = codesLayout(n, dimOrNull != null ? dim : values.length);
  const codes = new Uint8Array(layout.rows * layout.width * 4);
  const at = (i: number) => (Math.floor(i / layout.perRow) * layout.width + (i % layout.perRow) * layout.texels) * 4;
  let vec: Int8Array | null = null;
  const t2 = performance.now();
  if (dimOrNull != null && r.vector) {
    const direct = dim % 4 === 0 && dim <= MAX_INPUTS;
    vec = direct ? new Int8Array(codes.buffer, 0, n * dim) : new Int8Array(n * dim);
    vec.set(r.vector.subarray(a * dim, b * dim));
    if (!direct) {
      const m = Math.min(dim, MAX_INPUTS);
      for (let c = 0; c < n; c++) for (let k = 0; k < m; k++) codes[at(c) + k] = vec[c * dim + k] & 255;
    }
  } else {
    const m = Math.min(values.length, MAX_INPUTS);
    for (let c = 0; c < n; c++) {
      const o = at(c);
      for (let k = 0; k < m; k++) codes[o + k] = values[k][c] & 255;
    }
  }
  const t3 = performance.now();
  const stats = withStats ? tileStats({ values, vector: vec, dim, ncell, n }, statsEvery) : null;
  const statsMs = performance.now() - t3;
  const packed = pack(cells, lod);
  const byteLength =
    cells.byteLength + ncell.byteLength + values.reduce((x, v) => x + v.byteLength, 0) + codes.byteLength +
    (vec && vec.buffer !== codes.buffer ? vec.byteLength : 0) +
    packed.starts.byteLength + packed.positions.byteLength + (packed.indices?.byteLength ?? 0);
  return {
    n, lod, cells, ncell, values, vector: vec, dim, codes,
    codesWidth: layout.width, codesRows: layout.rows, texels: layout.texels, perRow: layout.perRow,
    ...packed, byteLength,
    fetched: timing.fetched, readMs: timing.readMs, decodeMs: timing.decodeMs + (t3 - t2),
    packMs: performance.now() - t3 - statsMs, stats, statsMs,
  };
}

/** Read tile `tile` (row group) of an open shard, or a piece of it, and pack it. */
export async function readTile(
  url: string, tile: number, lod: number, columns: string[], withStats = false, vector: { name: string; dim: number } | null = null,
  opts: ReadOptions = {},
): Promise<PackedTile> {
  const s = shards.get(url);
  if (!s) throw new Error(`shard not open: ${url}`);
  const n = s.rowStarts[tile + 1] - s.rowStarts[tile];
  const dim = vector ? vector.dim : null;
  if (opts.pieceRows && opts.piece != null) {
    // a piece samples its PCA rows as densely as a 16,384-row tile would
    // (whatever its row group's size)
    const statsEvery = opts.statsEvery ?? Math.max(1, Math.ceil(Math.min(n, STATS_TILE) / STATS_ROWS));
    opts = { ...opts, statsEvery, piece: opts.piece, pieceRows: opts.pieceRows };
    const piece = opts.piece!, pieceRows = opts.pieceRows!;
    const a = piece * pieceRows, b = Math.min(n, a + pieceRows);
    if (opts.pages) {
      const r = await readPiecePages(s, url, tile, piece, pieceRows, columns, vector);
      return packRows(r, 0, r.n, lod, dim, withStats, statsEvery, r);
    }
    // the row group read once a worker; the first piece carries its cost
    let first = false;
    const r = await cached(decoded, `${url}#${tile}`, DECODED_KEEP, () => ((first = true), readRows(s, tile, columns, vector, opts)));
    return packRows(r, a, b, lod, dim, withStats, statsEvery, first ? r : { fetched: 0, readMs: 0, decodeMs: 0 });
  }
  const r = await readRows(s, tile, columns, vector, opts);
  return packRows(r, 0, n, lod, dim, withStats, opts.statsEvery, r);
}

/**
 * Boxes of a row group's pieces of `pieceRows` rows, from the centres of
 * every `sample`-th cell, widened by `sample` cells and one more: the cells
 * between two samples lie along the A5 curve, within that of them.
 */
export async function probeTile(url: string, tile: number, lod: number, pieceRows: number, sample = 4): Promise<{ boxes: Float32Array; fetched: number; ms: number }> {
  const s = shards.get(url);
  if (!s) throw new Error(`shard not open: ${url}`);
  const t0 = performance.now();
  let fetched = 0;
  const n = s.rowStarts[tile + 1] - s.rowStarts[tile];
  const cells: bigint[] = [];
  await parquetRead({
    file: counted(s, (b) => (fetched += b)), metadata: s.metadata, compressors,
    rowStart: s.rowStarts[tile], rowEnd: s.rowStarts[tile + 1], columns: ["cell"],
    onChunk: (c) => {
      const d = c.columnData as ArrayLike<unknown>;
      for (let i = 0; i < d.length; i++) cells.push(typeof d[i] === "bigint" ? BigInt.asUintN(64, d[i] as bigint) : BigInt(d[i] as number));
    },
  });
  const pieces = Math.ceil(n / pieceRows);
  const boxes = new Float32Array(pieces * 4);
  const pad = (sample + 1) * cellSize(lod);
  for (let p = 0; p < pieces; p++) {
    let w = Infinity, so = Infinity, e = -Infinity, no = -Infinity;
    const a = p * pieceRows, b = Math.min(n, a + pieceRows);
    for (let i = a; i < b; i += sample) {
      const [lon, lat] = cellToLonLat(cells[i]);
      w = Math.min(w, lon); e = Math.max(e, lon); so = Math.min(so, lat); no = Math.max(no, lat);
    }
    const [lon, lat] = cellToLonLat(cells[b - 1]);
    w = Math.min(w, lon); e = Math.max(e, lon); so = Math.min(so, lat); no = Math.max(no, lat);
    const dLat = pad / 110574, dLon = dLat / Math.max(0.05, Math.cos(((so + no) / 2) * Math.PI / 180));
    boxes.set([w - dLon, so - dLat, e + dLon, no + dLat], p * 4);
  }
  return { boxes, fetched, ms: performance.now() - t0 };
}

function copyInto<T>(out: { [i: number]: T; length: number }, parts: ArrayLike<unknown>[], conv: (v: unknown) => T) {
  let k = 0;
  for (const p of parts) for (let i = 0; i < p.length && k < out.length; i++) out[k++] = conv(p[i]);
  if (k !== out.length) throw new Error(`column has ${k} values for ${out.length} rows`);
}

/** Boundaries (a5-js, its default edge segments, open rings) and fans. */
export function pack(cells: BigUint64Array, lod: number) {
  const n = cells.length;
  const rings = new Array<[number, number][]>(n);
  let nv = 0;
  for (let i = 0; i < n; i++) {
    const r = cellToBoundary(cells[i], { closedRing: false }) as [number, number][];
    rings[i] = r;
    nv += r.length;
  }
  const starts = new Uint32Array(n + 1);
  const positions = new Float64Array(nv * 2);
  const fan = lod >= FAN_MIN_LOD;
  const indices = fan ? new Uint32Array(3 * (nv - 2 * n)) : null;
  let v = 0;
  let k = 0;
  for (let i = 0; i < n; i++) {
    starts[i] = v;
    const r = rings[i];
    for (let j = 0; j < r.length; j++) {
      positions[2 * (v + j)] = r[j][0];
      positions[2 * (v + j) + 1] = r[j][1];
    }
    if (indices) {
      for (let j = 1; j + 1 < r.length; j++) {
        indices[k++] = v;
        indices[k++] = v + j;
        indices[k++] = v + j + 1;
      }
    }
    v += r.length;
  }
  starts[n] = v;
  return { starts, positions, indices };
}

/** The transferable buffers of a packed tile. */
export function transferables(t: PackedTile): ArrayBuffer[] {
  const out = [t.cells.buffer, t.ncell.buffer, t.starts.buffer, t.positions.buffer, t.codes.buffer, ...t.values.map((v) => v.buffer)];
  // (a buffer listed twice fails the transfer)
  if (t.vector && t.vector.buffer !== t.codes.buffer) out.push(t.vector.buffer);
  if (t.indices) out.push(t.indices.buffer);
  if (t.stats) out.push(t.stats.sum.buffer, t.stats.cross.buffer);
  return out as ArrayBuffer[];
}

/** A message to a decode worker. */
export type WorkerRequest =
  | { id: number; type: "index"; url: string; bytes: number; footer: ArrayBuffer }
  | { id: number; type: "probe"; url: string; bytes: number; footer?: ArrayBuffer; tile: number; lod: number; pieceRows: number }
  | {
      id: number; type: "decode"; url: string; bytes: number; footer?: ArrayBuffer; tile: number; lod: number;
      columns: string[]; stats?: boolean; vector?: { name: string; dim: number } | null; read?: ReadOptions;
    };

/**
 * What a worker does with a message: open a shard (its index, and each
 * tile's rows and compressed bytes, for the view budget) or read a tile.
 * The reply echoes the id, with a result or an error.
 */
export async function handleRequest(m: WorkerRequest): Promise<{ reply: Record<string, unknown>; transfer: ArrayBuffer[] }> {
  try {
    if (m.type === "index") {
      const md = openShard(m.url, m.bytes, m.footer);
      const tileRows = md.row_groups.map((rg) => Number(rg.num_rows));
      const tileBytes = md.row_groups.map((rg) => rg.columns.reduce((a, c) => a + Number(c.meta_data?.total_compressed_size ?? 0), 0));
      return { reply: { id: m.id, kv: keyValues(md), rowGroups: md.row_groups.length, tileRows, tileBytes }, transfer: [] };
    }
    if (!hasShard(m.url)) {
      if (!m.footer) throw new Error(`no footer for ${m.url}`);
      openShard(m.url, m.bytes, m.footer);
    }
    if (m.type === "probe") {
      const p = await probeTile(m.url, m.tile, m.lod, m.pieceRows);
      return { reply: { id: m.id, probe: p }, transfer: [p.boxes.buffer as ArrayBuffer] };
    }
    const tile = await readTile(m.url, m.tile, m.lod, m.columns, m.stats ?? false, m.vector ?? null, m.read ?? {});
    return { reply: { id: m.id, tile }, transfer: transferables(tile) };
  } catch (e) {
    return { reply: { id: m.id, error: String((e as Error)?.message ?? e) }, transfer: [] };
  }
}
