import { beforeAll, describe, expect, it } from "vitest";
import { contains, lonLatToCell, polygonToCells, union } from "a5-js";
import { PyramidSource, WORLD } from "../src/source.js";
import { coveringRanges, markRows, rangesMeet, regionRanges, type Ring } from "../src/region.js";
import type { Box } from "../src/format.js";
import type { PackedTile } from "../src/tile-core.js";
import { ORIGIN, installFetch, localWorker, served } from "./helpers.js";

beforeAll(installFetch);

type P = Parameters<typeof polygonToCells>[0];
/** a5-js's covering of rings at a level, the reference the ranges are checked against. */
const cover = (rings: Ring[], lod: number) =>
  rings.map((r) => polygonToCells(r as unknown as P, lod, { containment: "overlapping" })).reduce((a, b) => union(a, b));

/** A seeded generator, so a failure repeats. */
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

/** The south-west half of a box as a triangle: its box is the whole box, so
 * only cell bounds can tell its tiles apart. */
function triangle(b: Box, pad = 0.0005): Ring {
  return [[b[0] - pad, b[1] - pad], [b[2] + pad, b[1] - pad], [b[0] - pad, b[3] + pad], [b[0] - pad, b[1] - pad]];
}

describe("a covering as id ranges", () => {
  const ring: Ring = [[117.79, 4.95], [117.81, 4.952], [117.802, 4.968], [117.795, 4.962]];
  const box: Box = [117.785, 4.945, 117.815, 4.973];

  it("holds the cells a5-js's covering contains, and no others", () => {
    const rand = rng(7);
    for (const lod of [6, 9, 12, 15, 17, 18]) {
      const cov = cover([ring], lod);
      const ranges = coveringRanges(cov);
      const cells = BigUint64Array.from(
        Array.from({ length: 3000 }, () => lonLatToCell([box[0] + rand() * (box[2] - box[0]), box[1] + rand() * (box[3] - box[1])] as never, lod)),
      );
      const want = Array.from(cells, (c) => (contains(cov, c) ? 1 : 0));
      // rows in id order, as a tile's are, and out of order
      const sorted = BigUint64Array.from([...cells].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)));
      const keepSorted = new Uint8Array(sorted.length);
      markRows(sorted, ranges, keepSorted);
      expect([...keepSorted]).toEqual(Array.from(sorted, (c) => (contains(cov, c) ? 1 : 0)));
      const keep = new Uint8Array(cells.length);
      markRows(cells, ranges, keep);
      expect([...keep]).toEqual(want);
      if (lod >= 12) expect(want.some((k) => k) && want.some((k) => !k)).toBe(true);
    }
  });

  it("unions rings, and tells whether a span of ids meets it", () => {
    const other: Ring = [[117.811, 4.965], [117.814, 4.965], [117.814, 4.97]];
    const lod = 16;
    const ranges = regionRanges([ring, other], lod);
    const cov = cover([ring, other], lod);
    const inside = lonLatToCell([117.8125, 4.966] as never, lod);
    expect(contains(cov, inside)).toBe(true);
    expect(rangesMeet(ranges, inside, inside)).toBe(true);
    const outside = lonLatToCell([117.79, 4.972] as never, lod);
    expect(contains(cov, outside)).toBe(false);
    expect(rangesMeet(ranges, outside, outside)).toBe(false);
    expect(rangesMeet(ranges, 0n, (1n << 64n) - 1n)).toBe(true);
  });
});

async function readLevel(s: PyramidSource, lod: number) {
  await s.ensure(s.tierOf(lod), WORLD);
  const refs = s.tilesIn(lod, WORLD);
  const tiles: PackedTile[] = await Promise.all(refs.map((r) => s.tile(r.key, lod)));
  return { refs, tiles };
}

const cellsOf = (tiles: PackedTile[]) => tiles.flatMap((t) => [...t.cells]).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

/** The leaves' box: the union of the boxes of the shards of the finest level. */
function leafBox(s: PyramidSource): Box {
  const lod = s.lods[s.lods.length - 1];
  const bs = s.set.shards.filter((e) => e.tier === s.tierOf(lod) && e.bbox).map((e) => e.bbox!);
  return [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))];
}

// vector: one tile a row group (row group statistics); store-v4: tiles as
// pages of larger row groups (the column index), one level a file
for (const [name, set, lods] of [
  ["vector", `${ORIGIN}vector/`, [12, 14, 16, 17, 18]],
  ["store-v4", `${ORIGIN}store-v4/`, [12, 15, 16, 17, 18]],
] as const) {
  describe(`a region over the ${name} set`, () => {
    const open = (opts = {}) => PyramidSource.open(set, { worker: localWorker, workers: 2, ...opts });

    it("draws exactly the cells of each level's covering, whole", async () => {
      const full = await open();
      const ring = triangle(leafBox(full));
      const s = await open({ region: [ring] });
      for (const lod of lods) {
        const all = cellsOf((await readLevel(full, lod)).tiles);
        const cov = cover([ring], lod);
        const want = all.filter((c) => contains(cov, c));
        const { tiles } = await readLevel(s, lod);
        expect(cellsOf(tiles), `LOD ${lod}`).toEqual(want);
        if (lod >= 16) expect(want.length).toBeLessThan(all.length);
        expect(want.length).toBeGreaterThan(0);
        // PCA sums over the region's rows alone
        for (const t of tiles) if (t.stats) expect(t.stats.w).toBe(t.ncell.reduce((a, b) => a + b, 0));
      }
      // a tile wholly outside is left unread; one that is read drops its rows outside
      expect(s.stats.tilesSkipped).toBeGreaterThan(0);
      expect(s.stats.rowsDropped).toBeGreaterThan(0);
      full.close();
      s.close();
    });

    it("leaves unread the tiles its cells' bounds rule out, where boxes cannot", async () => {
      const full = await open();
      const ring = triangle(leafBox(full));
      const s = await open({ region: [ring] });
      const lod = 18;
      await full.ensure(full.tierOf(lod), WORLD);
      await s.ensure(s.tierOf(lod), WORLD);
      // the triangle's box holds every leaf tile's box: only cell bounds skip
      const all = full.tilesIn(lod, WORLD);
      const read = s.tilesIn(lod, WORLD);
      expect(read.length).toBeLessThan(all.length);
      const before = served.length;
      await Promise.all(read.map((r) => s.tile(r.key, lod)));
      const n = served.length - before;
      const before2 = served.length;
      await Promise.all(all.map((r) => full.tile(r.key, lod)));
      expect(n).toBeLessThan(served.length - before2);
      full.close();
      s.close();
    });
  });
}

describe("the region's shards and options", () => {
  it("leaves a shard outside the region's box unopened", async () => {
    const s = await PyramidSource.open(`${ORIGIN}vector/`, { worker: localWorker, workers: 1, region: [[[0, 0], [1, 0], [1, 1]]] });
    await s.ensure(s.tierOf(18), WORLD);
    expect(s.tilesIn(18, WORLD)).toEqual([]);
    expect(s.stats.shardsOpen).toBe(0);
    s.close();
  });

  it("refuses a ring that is not one", async () => {
    await expect(PyramidSource.open(`${ORIGIN}vector/`, { worker: localWorker, region: [[[0, 0], [1, 1]]] })).rejects.toThrow(/three points/);
  });

  it("merges reads by default only off http(s)", async () => {
    const s = await PyramidSource.open(`${ORIGIN}vector/`, { worker: localWorker });
    expect(s.merge).toBe(false);
    s.close();
    const m = await PyramidSource.open(`${ORIGIN}vector/`, { worker: localWorker, merge: true });
    expect(m.merge).toBe(true);
    m.close();
    const set = { ...m.set, shards: m.set.shards.map((e) => ({ ...e, url: undefined, path: e.path })) };
    const local = PyramidSource.fromSet(set, { worker: localWorker, base: "file:///data/vector/" });
    expect(local.merge).toBe(true);
    local.close();
  });

  it("draws every cell under two rings that share an edge", async () => {
    const full = await PyramidSource.open(`${ORIGIN}vector/`, { worker: localWorker, workers: 1 });
    const b = leafBox(full);
    const mid = (b[0] + b[2]) / 2;
    const west: Ring = [[b[0] - 0.001, b[1] - 0.001], [mid, b[1] - 0.001], [mid, b[3] + 0.001], [b[0] - 0.001, b[3] + 0.001]];
    const east: Ring = [[mid, b[1] - 0.001], [b[2] + 0.001, b[1] - 0.001], [b[2] + 0.001, b[3] + 0.001], [mid, b[3] + 0.001]];
    const both = await PyramidSource.open(`${ORIGIN}vector/`, { worker: localWorker, workers: 1, region: [west, east] });
    const lod = 18;
    expect(cellsOf((await readLevel(both, lod)).tiles)).toEqual(cellsOf((await readLevel(full, lod)).tiles));
    full.close();
    both.close();
  });
});
