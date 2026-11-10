import { describe, expect, it } from "vitest";
import { ScreenPca, pcaBasis, type PcaView } from "../src/pca.js";
import { tileStats, type PackedTile } from "../src/tile-core.js";

/** A tile-like object with sums over `rows` (each d codes). */
function tile(rows: number[][], first = 1n): PackedTile {
  const n = rows.length, d = rows[0].length;
  const vector = new Int8Array(rows.flat());
  const stats = tileStats({ values: [], vector, dim: d, ncell: new Uint32Array(n).fill(1), n });
  return { n, cells: new BigUint64Array([first]), stats } as unknown as PackedTile;
}

// codes spread along (1, 1, 0, 0) far more than along (0, 0, 1, -1)
function rows(seed: number, n = 200): number[][] {
  let s = seed;
  const r = () => ((s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31) * 2 - 1;
  return Array.from({ length: n }, () => {
    const a = 100 * r(), b = 20 * r();
    return [a, a, b, -b].map(Math.round);
  });
}

describe("pcaBasis", () => {
  it("finds the axis of greatest variance", () => {
    const b = pcaBasis([tile(rows(1)), tile(rows(2))])!;
    const first = b.rank.indexOf(1);
    const axis = Array.from(b.axes.subarray(first * 4, first * 4 + 4));
    expect(Math.abs(axis[0])).toBeCloseTo(Math.SQRT1_2, 1);
    expect(Math.abs(axis[1])).toBeCloseTo(Math.SQRT1_2, 1);
    expect(b.variance[first]).toBeGreaterThan(b.variance[b.rank.indexOf(2)]);
  });
});

describe("ScreenPca", () => {
  const level = (n: number): PackedTile[] => Array.from({ length: n }, (_, i) => tile(rows(10 + i), BigInt(i + 1)));
  const back = level(2);
  const view = (tiles: PackedTile[], backdropReady = true): PcaView => ({ lod: 14, tiles, backdropLod: 12, backdrop: back, backdropReady });

  it("starts from the backdrop, which stands while complete, and settles on every tile of the level", () => {
    const p = new ScreenPca();
    expect(p.update({ lod: 14, tiles: [], backdropLod: 12, backdrop: [], backdropReady: false }, false)).toBeNull();
    // the backdrop's tiles while the level has none
    expect(p.update(view([], false), false)!.basis.tiles).toBe(2);
    // the level's first tiles while the backdrop is complete: it stands
    expect(p.update(view(level(3)), false)).toBeNull();
    expect(p.update(view(level(9)), true)!.basis.tiles).toBe(9);
  });

  it("without a backdrop, redoes a provisional basis as the level's tiles double", () => {
    const p = new ScreenPca();
    const bare = (tiles: PackedTile[]): PcaView => ({ lod: 14, tiles, backdropLod: null, backdrop: [], backdropReady: false });
    expect(p.update(bare(level(3)), false)!.basis.tiles).toBe(3);
    expect(p.update(bare(level(5)), false)).toBeNull();
    expect(p.update(bare(level(6)), false)!.basis.tiles).toBe(6);
  });

  it("keeps its colours while locked", () => {
    const p = new ScreenPca();
    p.locked = true;
    expect(p.update(view(level(3)), true)).toBeNull();
  });
});
