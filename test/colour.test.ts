// The colourings against the formulas they replace, through the arithmetic
// the vertex shader does (codes-layer.ts): offset + weights . inputs, the
// inputs codes or their AEF values, clamped to 0..1.

import { beforeAll, describe, expect, it } from "vitest";
import { WEIGHT_VEC4S, bandColouring, neutralColouring, weightIndex, type Colouring } from "../src/colour.js";
import { pcaBasis, pcaColouring } from "../src/pca.js";
import { MAX_INPUTS, keyValues, openShard, readTile } from "../src/tile-core.js";
import type { PackedTile } from "../src/tile-core.js";
import type { PyramidIndex } from "../src/format.js";
import { ORIGIN, decode, footerOf, installFetch, readSet } from "./helpers.js";

beforeAll(installFetch);

/** The shader's colour for one cell's codes, 0..255 per channel. */
function shade(c: Colouring, codes: ArrayLike<number>): number[] {
  return [0, 1, 2].map((ch) => {
    let v = c.offset[ch];
    for (let k = 0; k < codes.length; k++) v += c.weights[weightIndex(ch, k)] * (c.decode ? decode(codes[k]) : codes[k]);
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  });
}

describe("weights", () => {
  it("are the shader's vec4 rows: channel ch, codes 4j to 4j + 3 at vec4 ch * 16 + j", () => {
    expect(WEIGHT_VEC4S).toBe(3 * (MAX_INPUTS / 4));
    expect(weightIndex(0, 0)).toBe(0);
    expect(weightIndex(0, 5)).toBe(4 + 1);
    expect(weightIndex(1, 0)).toBe(16 * 4);
    expect(weightIndex(2, 63)).toBe((32 + 15) * 4 + 3);
  });
});

describe("bandColouring", () => {
  it("stretches each band linearly between its two codes, as the CPU table did", () => {
    const ranges: [number, number][] = [[-43, 69], [-60, 34], [22, 72]];
    const c = bandColouring({ ranges, version: 0 });
    let worst = 0;
    for (let code = -128; code < 128; code++) {
      const got = shade(c, [code, code, code]);
      ranges.forEach(([lo, hi], ch) => {
        const want = Math.round(Math.min(1, Math.max(0, (code - lo) / (hi - lo))) * 255);
        worst = Math.max(worst, Math.abs(got[ch] - want));
      });
    }
    expect(worst).toBeLessThanOrEqual(1);
    expect(c.offset[3]).toBe(1);
  });

  it("gives every cell one grey while a view has no basis", () => {
    expect(shade(neutralColouring(), [10, -50, 90])).toEqual([150, 150, 150]);
  });
});

describe("pcaColouring", () => {
  it("maps each channel's score over the mean +/- 2 sd, as the CPU pass did", async () => {
    const set = readSet("vector");
    const tiles: PackedTile[] = [];
    for (const sh of set.shards.filter((s) => s.tier === 2)) {
      const url = `${ORIGIN}vector/${sh.path}`;
      const index = JSON.parse(keyValues(openShard(url, sh.bytes, footerOf("vector", sh))).a5pyramid) as PyramidIndex;
      const level = index.levels.find((l) => l.lod === 18)!;
      for (let i = level.tiles[0]; i < level.tiles[1]; i++) tiles.push(await readTile(url, i, 18, [], true, { name: "embedding", dim: 64 }));
    }
    const b = pcaBasis(tiles)!;
    const c = pcaColouring(b);
    expect(c.decode).toBe(true);
    let worst = 0, cells = 0;
    for (const t of tiles) {
      for (let i = 0; i < t.n; i++) {
        const codes = t.vector!.subarray(i * 64, (i + 1) * 64);
        const got = shade(c, codes);
        for (let ch = 0; ch < 3; ch++) {
          let s = 0;
          for (let k = 0; k < 64; k++) s += b.axes[ch * 64 + k] * (decode(codes[k]) - b.mean[k]);
          const want = Math.round(Math.min(255, Math.max(0, 127.5 + (127.5 * s) / (2 * Math.sqrt(b.variance[ch])))));
          worst = Math.max(worst, Math.abs(got[ch] - want));
        }
        cells++;
      }
    }
    expect(cells).toBe(1011);
    expect(worst).toBeLessThanOrEqual(1);
  });
});
