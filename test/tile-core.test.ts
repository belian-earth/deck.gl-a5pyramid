import { beforeAll, describe, expect, it } from "vitest";
import { AEF_DECODE, FAN_MIN_LOD, codesLayout, keyValues, openShard, readTile } from "../src/tile-core.js";
import type { PyramidIndex } from "../src/format.js";
import { ORIGIN, footerOf, installFetch, readSet } from "./helpers.js";

beforeAll(installFetch);

for (const name of ["bands", "vector"]) {
  describe(`tiles of the ${name} fixture`, () => {
    it("decode to the rows, order, geometry, codes and sums the footer promises", async () => {
      const set = readSet(name);
      const v = set.columns.find((c) => c.dim != null);
      const vector = v ? { name: v.name, dim: v.dim! } : null;
      const columns = set.columns.filter((c) => c.dim == null).map((c) => c.name);
      const d = vector ? vector.dim : columns.length;
      let tiles = 0;
      let wrong = 0;
      for (const sh of set.shards) {
        const url = `${ORIGIN}${name}/${sh.path}`;
        const md = openShard(url, sh.bytes, footerOf(name, sh));
        const index = JSON.parse(keyValues(md).a5pyramid) as PyramidIndex;
        for (const level of index.levels) {
          for (let i = level.tiles[0]; i < level.tiles[1]; i++) {
            const t = await readTile(url, i, level.lod, columns, true, vector);
            tiles++;
            expect(t.n).toBe(Number(md.row_groups[i].num_rows));
            expect(t.lod).toBe(level.lod);
            for (let k = 1; k < t.n; k++) if (!(t.cells[k] > t.cells[k - 1])) wrong++;
            // open rings, one fan a cell where deck would otherwise earcut
            expect(t.starts.length).toBe(t.n + 1);
            expect(t.positions.length).toBe(2 * t.starts[t.n]);
            if (level.lod >= FAN_MIN_LOD) expect(t.indices!.length).toBe(3 * (t.starts[t.n] - 2 * t.n));
            else expect(t.indices).toBeNull();
            // the codes texture: each cell's codes in consecutive bytes of one row
            const L = codesLayout(t.n, d);
            expect([t.texels, t.perRow, t.codesWidth, t.codesRows]).toEqual([L.texels, L.perRow, L.width, L.rows]);
            expect(t.codes.length).toBe(L.rows * L.width * 4);
            for (let c = 0; c < t.n; c++) {
              const o = (Math.floor(c / L.perRow) * L.width + (c % L.perRow) * L.texels) * 4;
              for (let k = 0; k < d; k++) {
                const code = vector ? t.vector![c * d + k] : t.values[k][c];
                if (t.codes[o + k] !== (code & 255)) wrong++;
              }
            }
            // a vector of whole texels is the texture's own bytes
            if (vector) expect(t.vector!.buffer).toBe(t.codes.buffer);
            // PCA sums, every row (the tiles are under STATS_ROWS)
            let w = 0;
            const sum = new Float64Array(d);
            for (let c = 0; c < t.n; c++) {
              w += t.ncell[c];
              for (let k = 0; k < d; k++) sum[k] += t.ncell[c] * AEF_DECODE[(vector ? t.vector![c * d + k] : t.values[k][c]) + 128];
            }
            expect(t.stats!.w).toBe(w);
            for (let k = 0; k < d; k++) expect(t.stats!.sum[k]).toBeCloseTo(sum[k], 6);
          }
        }
      }
      expect(wrong).toBe(0);
      expect(tiles).toBe(set.shards.reduce((a, s) => a + s.tiles, 0));
    });
  });
}
