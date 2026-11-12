// A vector stored as a fixed-length list column (LIST<INT8> of `dim`
// elements a row, no nulls; store schema v4), read without hyparquet's row
// assembly. parquetRead decodes dictionary indices into a JS array and then
// builds one JS array a row; on a 16,384-row AEF tile that is 40 to 50 ms.
// Here the column chunk's pages are walked with hyparquet's own primitives
// and every element is written into one Int8Array, rows by `dim`, which can
// be the tile's codes texture itself: 5 to 7 ms a tile with a wasm zstd,
// about what a fixed binary column costs (measured 2026-11-12, Node 20).

import type { ColumnMetaData, Compressors, FileMetaData } from "hyparquet";
import { parquetHeader } from "hyparquet/src/column.js";
import { decompressPage } from "hyparquet/src/datapage.js";
import { readRleBitPackedHybrid } from "hyparquet/src/encoding.js";
import { readPlain } from "hyparquet/src/plain.js";
import { getMaxDefinitionLevel, getMaxRepetitionLevel, getSchemaPath } from "hyparquet/src/schema.js";

/** A vector column stored as a list: its chunk's metadata in a row group,
 * and its nesting levels. */
export interface ListColumn {
  index: number;
  maxDef: number;
  maxRep: number;
}

/** The list layout of vector column `name`, or null when the column is not a
 * list of INT32-backed integers (a fixed binary vector, say). */
export function listColumn(metadata: FileMetaData, name: string): ListColumn | null {
  const cols = metadata.row_groups[0]?.columns ?? [];
  const index = cols.findIndex((c) => c.meta_data?.path_in_schema[0] === name);
  const meta = index >= 0 ? cols[index].meta_data : undefined;
  if (!meta || meta.path_in_schema.length < 2 || meta.type !== "INT32") return null;
  const path = getSchemaPath(metadata.schema, meta.path_in_schema);
  const maxRep = getMaxRepetitionLevel(path);
  if (maxRep !== 1) return null;
  return { index, maxDef: getMaxDefinitionLevel(path), maxRep };
}

/** The byte range of a column chunk: its dictionary page, if any, then its
 * data pages. */
export function chunkRange(meta: ColumnMetaData): [number, number] {
  const start = Number(meta.dictionary_page_offset ?? meta.data_page_offset);
  return [start, start + Number(meta.total_compressed_size)];
}

/**
 * Dictionary indices in the RLE/bit-packed hybrid encoding, from
 * `bytes[off, end)`, looked up in `dict` and written to `dst`. Width 8 (a
 * dictionary of 129 to 256 codes, as AEF's) reads bit-packed runs as bytes.
 */
export function decodeDict(bytes: Uint8Array, off: number, end: number, width: number, dict: Int8Array, dst: Int8Array): void {
  const n = dst.length;
  if (width === 0) {
    dst.fill(dict[0]);
    return;
  }
  const mask = (1 << width) - 1;
  let i = 0;
  while (i < n && off < end) {
    let header = 0;
    let shift = 0;
    let b: number;
    do {
      b = bytes[off++];
      header |= (b & 0x7f) << shift;
      shift += 7;
    } while (b & 0x80);
    if (header & 1) {
      const count = (header >>> 1) * 8;
      const m = Math.min(count, n - i);
      if (width === 8) {
        for (let k = 0; k < m; k++) dst[i + k] = dict[bytes[off + k]];
        off += count;
      } else {
        let buf = 0;
        let bits = 0;
        for (let k = 0; k < count; k++) {
          while (bits < width) {
            buf |= bytes[off++] << bits;
            bits += 8;
          }
          if (k < m) dst[i + k] = dict[buf & mask];
          buf >>>= width;
          bits -= width;
        }
      }
      i += m;
    } else {
      const count = header >>> 1;
      let v = 0;
      for (let k = 0, w = (width + 7) >> 3; k < w; k++) v |= bytes[off++] << (8 * k);
      dst.fill(dict[v], i, Math.min(n, i + count));
      i += count;
    }
  }
  if (i < n) throw new Error(`dictionary indices end at ${i} of ${n}`);
}

/** Walk RLE/bit-packed hybrid runs without decoding them; a callback
 * returning false rejects. */
function walkRuns(
  bytes: Uint8Array, width: number, n: number,
  rle: (i: number, count: number, value: number) => boolean,
  packed: (i: number, count: number, off: number) => boolean,
): boolean {
  let o = 0;
  let i = 0;
  const w = (width + 7) >> 3;
  while (i < n && o < bytes.length) {
    let header = 0;
    let shift = 0;
    let b: number;
    do {
      b = bytes[o++];
      header |= (b & 0x7f) << shift;
      shift += 7;
    } while (b & 0x80);
    if (header & 1) {
      const groups = header >>> 1;
      if (!packed(i, Math.min(groups * 8, n - i), o)) return false;
      o += groups * width;
      i += groups * 8;
    } else {
      const count = header >>> 1;
      let v = 0;
      for (let k = 0; k < w; k++) v |= bytes[o++] << (8 * k);
      if (!rle(i, Math.min(count, n - i), v)) return false;
      i += count;
    }
  }
  return i >= n;
}

/**
 * Whether a page's levels are whole rows of `dim` with no nulls, read off
 * the runs: definition levels all `maxDef`, repetition levels 0 at each row
 * start and 1 elsewhere, bit-packed repetition runs compared byte by byte
 * with that pattern. False only when a run shows otherwise; null when a run
 * is not of a shape this reads, and the levels must be decoded.
 */
export function levelsWhole(rep: Uint8Array, def: Uint8Array, n: number, dim: number, maxDef: number): boolean | null {
  let shape = true;
  const defOk = !maxDef || walkRuns(def, 32 - Math.clz32(maxDef), n, (_i, _c, v) => v === maxDef, () => (shape = false));
  if (!defOk) return shape ? false : null;
  const repOk = walkRuns(rep, 1, n,
    (i, c, v) => (v === 1 ? Math.ceil(i / dim) * dim >= i + c : c === 1 && i % dim === 0),
    (i, c, off) => {
      if (i % 8 || dim % 8 || c % 8) return (shape = false);
      for (let j = 0, q = i; j < c / 8; j++, q += 8) if (rep[off + j] !== (q % dim === 0 ? 0xfe : 0xff)) return false;
      return true;
    });
  return repOk ? true : shape ? false : null;
}

/** Decode levels and check them, the slow way. */
function checkLevels(rep: Uint8Array, def: Uint8Array, n: number, dim: number, maxDef: number): void {
  const read = (bytes: Uint8Array, width: number) => {
    const out = new Uint8Array(n);
    readRleBitPackedHybrid({ view: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), offset: 0 }, width, out, bytes.byteLength);
    return out;
  };
  const r = read(rep, 1);
  const d = maxDef ? read(def, 32 - Math.clz32(maxDef)) : null;
  for (let i = 0; i < n; i++) {
    if (r[i] !== (i % dim === 0 ? 0 : 1)) throw new Error(`a row of other than ${dim} elements at value ${i}`);
    if (d && d[i] !== maxDef) throw new Error(`a null or empty list at value ${i}`);
  }
}

/**
 * Read a fixed-length list column chunk into `out` (rows × `dim` codes).
 * Pages may be v1 or v2, dictionary or plain encoded. Every page is checked
 * to hold whole rows of `dim` with no nulls, from its v2 header or its
 * level runs, else by decoding its levels.
 */
export function readFixedList(
  chunk: Uint8Array, meta: ColumnMetaData, list: ListColumn, dim: number, out: Int8Array, compressors?: Compressors,
): Int8Array {
  const total = Number(meta.num_values);
  if (total !== out.length) throw new Error(`the chunk holds ${total} values for ${out.length / dim} rows of ${dim}`);
  const reader = { view: new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength), offset: 0 };
  let dict: Int8Array | null = null;
  let pos = 0;
  while (pos < total && reader.offset < chunk.byteLength) {
    const h = parquetHeader(reader);
    const body = chunk.subarray(reader.offset, reader.offset + h.compressed_page_size);
    reader.offset += h.compressed_page_size;
    if (h.type === "DICTIONARY_PAGE") {
      const page = decompressPage(body, Number(h.uncompressed_page_size), meta.codec, compressors);
      const n = h.dictionary_page_header?.num_values ?? 0;
      dict = Int8Array.from(readPlain({ view: new DataView(page.buffer, page.byteOffset, page.byteLength), offset: 0 }, "INT32", n, undefined) as Int32Array);
      continue;
    }
    let n: number;
    let encoding: string;
    let values: Uint8Array;
    let rep: Uint8Array;
    let def: Uint8Array;
    let checked = false;
    if (h.type === "DATA_PAGE") {
      const dp = h.data_page_header!;
      n = dp.num_values;
      encoding = dp.encoding;
      const page = decompressPage(body, Number(h.uncompressed_page_size), meta.codec, compressors);
      const v = new DataView(page.buffer, page.byteOffset, page.byteLength);
      let o = 0;
      const rl = v.getUint32(o, true);
      rep = page.subarray(o + 4, o + 4 + rl);
      o += 4 + rl;
      if (list.maxDef) {
        const dl = v.getUint32(o, true);
        def = page.subarray(o + 4, o + 4 + dl);
        o += 4 + dl;
      } else def = new Uint8Array(0);
      values = page.subarray(o);
    } else if (h.type === "DATA_PAGE_V2") {
      const dp = h.data_page_header_v2!;
      n = dp.num_values;
      encoding = dp.encoding;
      if (dp.num_nulls || dp.num_rows * dim !== n) throw new Error(`a page of ${dp.num_rows} rows holds ${n} values with ${dp.num_nulls} nulls`);
      checked = true;
      const rl = dp.repetition_levels_byte_length;
      const dl = dp.definition_levels_byte_length;
      rep = body.subarray(0, rl);
      def = body.subarray(rl, rl + dl);
      values = body.subarray(rl + dl);
      const size = Number(h.uncompressed_page_size) - rl - dl;
      if (dp.is_compressed !== false && size) values = decompressPage(values, size, meta.codec, compressors);
    } else throw new Error(`unsupported page type ${h.type}`);
    if (pos % dim || n % dim) throw new Error("a page holds part of a row");
    if (!checked) {
      const whole = levelsWhole(rep, def, n, dim, list.maxDef);
      if (whole === false) throw new Error(`a page's levels are not rows of ${dim} without nulls`);
      if (whole === null) checkLevels(rep, def, n, dim, list.maxDef);
    }
    const dst = out.subarray(pos, pos + n);
    if (encoding === "RLE_DICTIONARY" || encoding === "PLAIN_DICTIONARY") {
      if (!dict) throw new Error("a dictionary-encoded page without a dictionary");
      decodeDict(values, 1, values.length, values[0], dict, dst);
    } else if (encoding === "PLAIN") {
      const v = readPlain({ view: new DataView(values.buffer, values.byteOffset, values.byteLength), offset: 0 }, "INT32", n, undefined) as Int32Array;
      for (let i = 0; i < n; i++) dst[i] = v[i];
    } else throw new Error(`unsupported encoding ${encoding} for a list vector`);
    pos += n;
  }
  if (pos !== total) throw new Error(`read ${pos} of ${total} values`);
  return out;
}
