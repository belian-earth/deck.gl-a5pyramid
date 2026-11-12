import { beforeAll, describe, expect, it } from "vitest";
import { PyramidSource, WORLD } from "../src/source.js";
import { locateTiles } from "../src/tile-core.js";
import { ORIGIN, installFetch, localWorker, readSet, served } from "./helpers.js";

beforeAll(installFetch);

async function until(f: () => boolean) {
  for (let i = 0; i < 5000 && !f(); i++) await new Promise((r) => setTimeout(r, 2));
}

const open = (name: string) => PyramidSource.open(`${ORIGIN}${name}/`, { worker: localWorker, workers: 2 });

describe("locateTiles", () => {
  it("cuts a row group into tiles of tile_size rows, the last shorter", () => {
    const t = locateTiles([4, 30, 7], { tile_size: 10, row_group_size: 30 });
    expect(t.map((x) => [x.rowGroup, x.page, x.start, x.rows, x.shared])).toEqual([
      [0, 0, 0, 4, false], [1, 0, 0, 10, true], [1, 1, 10, 10, true], [1, 2, 20, 10, true], [2, 0, 0, 7, false],
    ]);
    // without a row group size each row group is a tile, whatever its rows
    expect(locateTiles([4, 30], { tile_size: 10 }).map((x) => x.rows)).toEqual([4, 30]);
  });
});

// The vector fixture in format version 4 (tools/fixture-vector-rg.R): row
// groups of up to 64 rows cut into tiles of 16, each a data page.
describe("row groups of several tiles", () => {
  it("hold every level's cells and codes as the one-tile-a-row-group set does", async () => {
    const whole = await open("vector");
    const paged = await open("vector-rg");
    expect(paged.set.format_version).toBe(4);
    let shared = 0;
    for (const lod of whole.lods) {
      for (const s of [whole, paged]) await s.ensure(s.tierOf(lod), WORLD);
      await until(() => paged.ready(lod, WORLD));
      const read = async (s: PyramidSource) => {
        const refs = s.tilesIn(lod, WORLD);
        const tiles = await Promise.all(refs.map((r) => s.tile(r.key, lod)));
        return { refs, tiles, cells: tiles.flatMap((t) => Array.from(t.cells)), vec: tiles.flatMap((t) => Array.from(t.vector!)) };
      };
      const a = await read(whole);
      const b = await read(paged);
      expect(b.cells).toEqual(a.cells);
      expect(b.vec).toEqual(a.vec);
      for (const [k, t] of b.tiles.entries()) {
        expect(t.n).toBeLessThanOrEqual(16);
        // each tile's box holds its cells' boundaries' first vertex
        const box = b.refs[k].box;
        for (let c = 0; c < t.n; c++) {
          const lon = t.positions[2 * t.starts[c]], lat = t.positions[2 * t.starts[c] + 1];
          expect(lon >= box[0] - 1e-3 && lon <= box[2] + 1e-3 && lat >= box[1] - 1e-3 && lat <= box[3] + 1e-3).toBe(true);
        }
      }
      shared += b.refs.filter((r) => (r.shard as any).tileGroup[r.tile] >= 0).length;
    }
    expect(shared).toBeGreaterThan(50);
    whole.close();
    paged.close();
  }, 60000);

  it("reads a tile by its pages, its row group's page index from the footer", async () => {
    const s = await open("vector-rg");
    await s.ensure(s.tierOf(18), WORLD);
    const refs = s.tilesIn(18, WORLD);
    const shard = refs[0].shard as any;
    const groups = new Map<number, typeof refs>();
    for (const r of refs) {
      const g = r.shard === shard ? shard.tileGroup[r.tile] : -1;
      if (g >= 0) groups.set(g, [...(groups.get(g) ?? []), r]);
    }
    // the third tile of a row group of four
    const ref = [...groups.values()].find((v) => v.length === 4)![2];
    const before = served.length;
    const t = await s.tile(ref.key, 18);
    // its pages and dictionary pages only: the page index came with the
    // footer, so no request reaches past the column chunks
    const entry = readSet("vector-rg").shards.find((x) => x.path === shard.entry.path)!;
    for (const r of served.slice(before)) {
      expect(Number(r.range!.slice(6).split("-")[0])).toBeLessThan(entry.bytes - entry.footer_bytes);
    }
    expect(t.n).toBe(16);
    // a fraction of its row group's bytes
    expect(t.fetched).toBeLessThan((shard.tileBytes[ref.tile] * 4) / 2);
    s.close();
  });
});
