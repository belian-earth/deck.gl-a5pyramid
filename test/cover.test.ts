// The base mask's premises: cell keys keep id order, a tile is a run of
// consecutive ids, and the finer cell at a point is found by a5-js.

import { beforeAll, describe, expect, it } from "vitest";
import { cellToLonLat, lonLatToCell } from "a5-js";
import { cellKey, inRuns } from "../src/codes-layer.js";
import { keyValues, openShard, readTile, type PackedTile } from "../src/tile-core.js";
import type { PyramidIndex } from "../src/format.js";
import { ORIGIN, footerOf, installFetch, readSet } from "./helpers.js";

beforeAll(installFetch);

async function level(lod: number): Promise<PackedTile[]> {
  const set = readSet("bands");
  const out: PackedTile[] = [];
  for (const sh of set.shards) {
    const url = `${ORIGIN}bands/${sh.path}`;
    const index = JSON.parse(keyValues(openShard(url, sh.bytes, footerOf("bands", sh))).a5pyramid) as PyramidIndex;
    const l = index.levels.find((x) => x.lod === lod);
    if (l) for (let i = l.tiles[0]; i < l.tiles[1]; i++) out.push(await readTile(url, i, lod, ["A", "B", "C"]));
  }
  return out;
}

describe("runs", () => {
  it("hold their own tile's cells and no other's", async () => {
    const tiles = await level(18);
    expect(tiles.length).toBeGreaterThan(5);
    const pairs = tiles.map((t) => [cellKey(t.cells[0]), cellKey(t.cells[t.n - 1])]).sort((a, b) => a[0] - b[0]);
    const runs = Float64Array.from(pairs.flat());
    for (let i = 1; i < pairs.length; i++) expect(pairs[i][0]).toBeGreaterThan(pairs[i - 1][1]);
    let outside = 0;
    for (const t of tiles) {
      const own = Float64Array.from([cellKey(t.cells[0]), cellKey(t.cells[t.n - 1])]);
      for (let i = 0; i < t.n; i++) {
        const k = cellKey(t.cells[i]);
        if (!inRuns(runs, k) || !inRuns(own, k)) outside++;
      }
      // keys keep the ids' order
      for (let i = 1; i < t.n; i++) expect(cellKey(t.cells[i])).toBeGreaterThan(cellKey(t.cells[i - 1]));
    }
    expect(outside).toBe(0);
    expect(inRuns(runs, pairs[0][0] - 1)).toBe(false);
    expect(inRuns(runs, pairs.at(-1)![1] + 1)).toBe(false);
  });

  it("are found from a coarser cell's centre", async () => {
    // the finer cell at each LOD-16 centre is a cell of LOD 18 near it, and
    // the fixture's leaves hold most such centres
    const coarse = await level(16);
    const fine = new Set((await level(18)).flatMap((t) => Array.from(t.cells, (c) => c)));
    let found = 0, all = 0;
    for (const t of coarse) {
      for (let i = 0; i < t.n; i++) {
        const [lon, lat] = cellToLonLat(t.cells[i]) as unknown as [number, number];
        const f = lonLatToCell([lon, lat] as never, 18);
        if (fine.has(f)) found++;
        all++;
      }
    }
    expect(found / all).toBeGreaterThan(0.8);
  });
});
