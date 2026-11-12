import { beforeAll, describe, expect, it } from "vitest";
import { decodeDict, levelsWhole } from "../src/fixed-list.js";
import { keyValues, openShard, readTile } from "../src/tile-core.js";
import type { PyramidIndex } from "../src/format.js";
import { ORIGIN, footerOf, installFetch, readSet } from "./helpers.js";

beforeAll(installFetch);

// vector-list is the vector fixture with its fixed binary vector rewritten
// as LIST<INT8>, row group for row group, by arrow.
describe("a vector stored as a list", () => {
  it("reads tile for tile as the same vector stored as fixed binary", async () => {
    const vector = { name: "embedding", dim: 64 };
    const a = readSet("vector");
    const b = readSet("vector-list");
    let tiles = 0;
    for (let s = 0; s < a.shards.length; s++) {
      const ua = `${ORIGIN}vector/${a.shards[s].path}`;
      const ub = `${ORIGIN}vector-list/${b.shards[s].path}`;
      const ma = openShard(ua, a.shards[s].bytes, footerOf("vector", a.shards[s]));
      openShard(ub, b.shards[s].bytes, footerOf("vector-list", b.shards[s]));
      const index = JSON.parse(keyValues(ma).a5pyramid) as PyramidIndex;
      for (const level of index.levels) {
        for (let i = level.tiles[0]; i < level.tiles[1]; i++) {
          const ta = await readTile(ua, i, level.lod, [], true, vector);
          const tb = await readTile(ub, i, level.lod, [], true, vector);
          expect(tb.n).toBe(ta.n);
          expect(tb.cells).toEqual(ta.cells);
          expect(tb.vector).toEqual(ta.vector);
          expect(tb.codes).toEqual(ta.codes);
          expect(tb.vector!.buffer).toBe(tb.codes.buffer);
          expect(tb.stats!.sum).toEqual(ta.stats!.sum);
          tiles++;
        }
      }
    }
    expect(tiles).toBe(a.shards.reduce((n, s) => n + s.tiles, 0));
  });
});

describe("dictionary indices", () => {
  it("decode bit-packed and run-length runs at any width", () => {
    const dict = Int8Array.from([-128, -3, 0, 7, 127]);
    // width 8: a bit-packed run of 8 bytes, then a run of 5 of index 4
    const w8 = Uint8Array.from([(1 << 1) | 1, 0, 1, 2, 3, 4, 0, 1, 2, 5 << 1, 4]);
    const out8 = new Int8Array(13);
    decodeDict(w8, 0, w8.length, 8, dict, out8);
    expect(Array.from(out8)).toEqual([-128, -3, 0, 7, 127, -128, -3, 0, 127, 127, 127, 127, 127]);
    // width 3: 8 values 0..4,0..2 packed little-endian into 3 bytes
    const vals = [0, 1, 2, 3, 4, 0, 1, 2];
    let bits = 0n;
    vals.forEach((v, k) => (bits |= BigInt(v) << BigInt(3 * k)));
    const w3 = Uint8Array.from([(1 << 1) | 1, Number(bits & 255n), Number((bits >> 8n) & 255n), Number((bits >> 16n) & 255n)]);
    const out3 = new Int8Array(6); // a run's padding beyond the values is dropped
    decodeDict(w3, 0, w3.length, 3, dict, out3);
    expect(Array.from(out3)).toEqual(vals.slice(0, 6).map((v) => dict[v]));
    expect(() => decodeDict(w3, 0, w3.length, 3, dict, new Int8Array(9))).toThrow(/end at 8 of 9/);
  });
});

describe("level runs", () => {
  const dim = 64;
  const n = 4 * dim;
  const def = Uint8Array.from([0x80, 0x04, 3]); // one run of 256 threes (header 512 as a varint)
  const packedRep = (broken: boolean) => {
    const rep = new Uint8Array(1 + n / 8);
    rep[0] = ((n / 8) << 1) | 1;
    for (let j = 0; j < n / 8; j++) rep[1 + j] = (j * 8) % dim === 0 ? 0xfe : 0xff;
    if (broken) rep[1 + 9] = 0xfe; // a row start inside a row: rows of 72 and 56
    return rep;
  };
  it("accept whole rows, bit-packed or as runs", () => {
    expect(levelsWhole(packedRep(false), def, n, dim, 3)).toBe(true);
    // arrow's shape: 8 bit-packed (0 then seven 1s), then a run of 56 ones, a row at a time
    const rows: number[] = [];
    for (let r = 0; r < 4; r++) rows.push((1 << 1) | 1, 0xfe, 56 << 1, 1);
    expect(levelsWhole(Uint8Array.from(rows), def, n, dim, 3)).toBe(true);
  });
  it("reject a row of other than dim elements, and a null", () => {
    expect(levelsWhole(packedRep(true), def, n, dim, 3)).toBe(false);
    // definition levels: 255 threes then a 2, an empty or null element
    expect(levelsWhole(packedRep(false), Uint8Array.from([0xfe, 0x03, 3, 0x02, 2]), n, dim, 3)).toBe(false);
  });
});

describe("a DuckDB list column", () => {
  it("reads delta-packed v2 pages, as DuckDB writes small row groups", async () => {
    const { readFileSync } = await import("node:fs");
    const { parquetMetadata } = await import("hyparquet");
    const { listColumn, chunkRange, readFixedList } = await import("../src/fixed-list.js");
    const b = readFileSync(new URL("./fixtures/lists/duckdb-v2.parquet", import.meta.url));
    const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    const md = parquetMetadata(ab);
    const list = listColumn(md, "v")!;
    expect(list).not.toBeNull();
    const meta = md.row_groups[0].columns[list.index].meta_data!;
    expect(meta.encodings).toContain("DELTA_BINARY_PACKED");
    const [s, e] = chunkRange(meta);
    const out = readFixedList(new Uint8Array(ab.slice(s, e)), meta, list, 64, new Int8Array(1000 * 64));
    // written as ((i * 37 + e * 11) % 256) - 128, row i, element e
    let wrong = 0;
    for (let i = 0; i < 1000; i++) for (let k = 0; k < 64; k++) if (out[i * 64 + k] !== ((i * 37 + k * 11) % 256) - 128) wrong++;
    expect(wrong).toBe(0);
  });
});
