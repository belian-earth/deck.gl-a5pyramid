import { beforeAll, describe, expect, it } from "vitest";
import { PyramidSource, WORLD } from "../src/source.js";
import { ORIGIN, installFetch, localWorker } from "./helpers.js";

beforeAll(installFetch);

async function until(f: () => boolean) {
  for (let i = 0; i < 5000 && !f(); i++) await new Promise((r) => setTimeout(r, 2));
}

// Probed pieces (tiles cut into runs of rows, boxes from their cells) must
// hold exactly the rows of the whole tiles, in order, with the same codes.
describe("probed pieces", () => {
  for (const set of ["vector", "vector-list"]) {
    it(`of the ${set} fixture rebuild every whole tile`, async () => {
      const whole = await PyramidSource.open(`${ORIGIN}${set}/`, { worker: localWorker, workers: 2, pieces: "off" });
      const cut = await PyramidSource.open(`${ORIGIN}${set}/`, { worker: localWorker, workers: 2, pieces: "probe", pieceRows: 16,
        read: { singleRange: true } });
      let compared = 0;
      for (const lod of whole.lods) {
        await whole.ensure(whole.tierOf(lod), WORLD);
        await cut.ensure(cut.tierOf(lod), WORLD);
        await until(() => cut.ready(lod, WORLD));
        expect(cut.ready(lod, WORLD)).toBe(true);
        const tiles = whole.tilesIn(lod, WORLD);
        const pieces = cut.tilesIn(lod, WORLD);
        for (const t of tiles) {
          const a = await whole.tile(t.key, lod);
          const mine = pieces.filter((p) => p.shard.entry.path === t.shard.entry.path && p.tile === t.tile)
            .sort((x, y) => x.piece! - y.piece!);
          expect(mine.length).toBe(Math.ceil(a.n / 16));
          const parts = await Promise.all(mine.map((p) => cut.tile(p.key, lod)));
          const cells = parts.flatMap((p) => Array.from(p.cells));
          expect(cells).toEqual(Array.from(a.cells));
          const vec = parts.flatMap((p) => Array.from(p.vector!));
          expect(vec).toEqual(Array.from(a.vector!));
          // each piece's box holds its cells' boundaries' first vertex
          for (let k = 0; k < parts.length; k++) {
            const b = cut.tileBox(mine[k].key)!;
            for (let c = 0; c < parts[k].n; c++) {
              const lon = parts[k].positions[2 * parts[k].starts[c]], lat = parts[k].positions[2 * parts[k].starts[c] + 1];
              expect(lon >= b[0] - 1e-3 && lon <= b[2] + 1e-3 && lat >= b[1] - 1e-3 && lat <= b[3] + 1e-3).toBe(true);
            }
          }
          compared++;
        }
      }
      expect(compared).toBeGreaterThan(10);
      whole.close();
      cut.close();
    }, 60000);
  }
});
