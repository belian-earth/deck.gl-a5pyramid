// Reading and packing tiles, in a worker (or under Node in tests): no DOM,
// no deck.gl. Ported from a5view's tileCore (lazy-renderer.js) for shards of
// a set: a shard's footer arrives as bytes fetched once by the main thread,
// and its tiles are read with range requests from the shard's URL.

import { parquetMetadata, parquetRead, type FileMetaData } from "hyparquet";
import { cellToBoundary } from "a5-js";

/** A tile ready for deck.gl: binary polygon attributes plus what hover needs. */
export interface PackedTile {
  n: number;
  lod: number;
  cells: BigUint64Array;
  ncell: Uint32Array;
  /** One per value column, the stored codes. */
  values: Int8Array[];
  /** Vertex start of each cell, n + 1 entries. */
  starts: Uint32Array;
  /** lon, lat per vertex (open rings). */
  positions: Float64Array;
  /** Fan triangles over each ring, absolute vertex indices; null to let deck triangulate. */
  indices: Uint32Array | null;
  /** Bytes the tile holds in memory, colours included; deck.gl's tile cache reads it. */
  byteLength: number;
  /** Bytes fetched to read the tile, and timings (ms). */
  fetched: number;
  readMs: number;
  packMs: number;
}

/** Cells at this LOD or coarser are triangulated by deck (earcut): their
 * segmented edges need not be convex in lon/lat. Finer ones get fans. */
export const FAN_MIN_LOD = 9;

interface Shard {
  metadata: FileMetaData;
  rowStarts: number[];
  file: RemoteFile;
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
  shards.set(url, { metadata, rowStarts, file: new RemoteFile(url, byteLength) });
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

/** Read tile `tile` (row group) of an open shard and pack it. */
export async function readTile(url: string, tile: number, lod: number, columns: string[]): Promise<PackedTile> {
  const s = shards.get(url);
  if (!s) throw new Error(`shard not open: ${url}`);
  const t0 = performance.now();
  // bytes fetched for this tile alone: reads of one shard run concurrently
  let fetched = 0;
  const file = {
    byteLength: s.file.byteLength,
    slice: async (start: number, end?: number) => {
      const buf = await s.file.slice(start, end);
      fetched += buf.byteLength;
      return buf;
    },
  };
  const chunks = new Map<string, ArrayLike<unknown>[]>();
  await parquetRead({
    file,
    metadata: s.metadata,
    rowStart: s.rowStarts[tile],
    rowEnd: s.rowStarts[tile + 1],
    columns: ["cell", "ncell", ...columns],
    onChunk: (c) => {
      const list = chunks.get(c.columnName) ?? [];
      list.push(c.columnData as ArrayLike<unknown>);
      chunks.set(c.columnName, list);
    },
  });
  const t1 = performance.now();
  const n = s.rowStarts[tile + 1] - s.rowStarts[tile];
  const cells = new BigUint64Array(n);
  copyInto(cells, chunks.get("cell") ?? [], (v) => (typeof v === "bigint" ? BigInt.asUintN(64, v) : BigInt(v as number)));
  const ncell = new Uint32Array(n);
  copyInto(ncell, chunks.get("ncell") ?? [], (v) => Number(v));
  const values = columns.map((name) => {
    const out = new Int8Array(n);
    copyInto(out, chunks.get(name) ?? [], (v) => (v == null ? 0 : Number(v)));
    return out;
  });
  const packed = pack(cells, lod);
  const byteLength =
    cells.byteLength + ncell.byteLength + values.reduce((a, v) => a + v.byteLength, 0) +
    packed.starts.byteLength + packed.positions.byteLength + (packed.indices?.byteLength ?? 0) +
    // per-vertex colours, built later on the main thread
    (packed.positions.length / 2) * 4;
  return {
    n,
    lod,
    cells,
    ncell,
    values,
    ...packed,
    byteLength,
    fetched,
    readMs: t1 - t0,
    packMs: performance.now() - t1,
  };
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
  const out = [t.cells.buffer, t.ncell.buffer, t.starts.buffer, t.positions.buffer, ...t.values.map((v) => v.buffer)];
  if (t.indices) out.push(t.indices.buffer);
  return out as ArrayBuffer[];
}
