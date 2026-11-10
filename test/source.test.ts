import { beforeAll, describe, expect, it } from "vitest";
import { PyramidSource, WORLD } from "../src/source.js";
import { BASE_BUDGET } from "../src/budget.js";
import { ORIGIN, installFetch, localWorker, readSet, served } from "./helpers.js";

beforeAll(installFetch);

const open = (name: string) => PyramidSource.open(ORIGIN + name + "/", { workers: 2, worker: localWorker });

describe("PyramidSource", () => {
  it("opens a set and its levels", async () => {
    const s = await open("bands");
    expect(s.lods).toEqual(Array.from({ length: 17 }, (_, i) => i + 2));
    expect(s.columns).toEqual(["A", "B", "C"]);
    expect(s.vector).toBeNull();
    const v = await open("vector");
    expect(v.vector).toEqual({ name: "embedding", dim: 64 });
    expect(v.pca).toBe(true);
    s.close();
    v.close();
  });

  it("knows a view's cost only once its tier's shards have opened, then exactly", async () => {
    const s = await open("bands");
    expect(s.costIn(18, WORLD)).toBeNull();
    await s.ensure(s.tierOf(18), WORLD);
    const c = s.costIn(18, WORLD)!;
    const refs = s.tilesIn(18, WORLD);
    expect(c.tiles).toBe(refs.length);
    // every leaf of the fixture, in its tiles
    expect(c.rows).toBe(1011);
    expect(c.bytes).toBeGreaterThan(0);
    // a box meeting one tile costs that tile alone
    const one = s.costIn(18, refs[0].box)!;
    expect(one.tiles).toBeGreaterThanOrEqual(1);
    expect(one.tiles).toBeLessThan(refs.length);
    s.close();
  });

  it("finds the base level: the finest of tier 0 within the budget", async () => {
    const s = await open("vector");
    expect(await s.baseLevel(BASE_BUDGET)).toBe(12);
    // the fixture's patch is one cell at LOD 9 and coarser
    expect(await s.baseLevel({ cells: 1, bytes: 1e9 })).toBe(9);
    expect(await s.baseLevel({ cells: 0, bytes: 1e9 })).toBe(null);
    s.close();
  });

  it("reads a tile once, then serves it from the cache", async () => {
    const s = await open("vector");
    await s.ensure(s.tierOf(17), WORLD);
    const [ref] = s.tilesIn(17, WORLD);
    const before = served.length;
    const t = await s.tile(ref.key, 17);
    const reads = served.length - before;
    expect(reads).toBeGreaterThan(0);
    expect(await s.tile(ref.key, 17)).toBe(t);
    expect(served.length - before).toBe(reads);
    expect(t.vector).not.toBeNull();
    expect(s.stats.tilesDecoded).toBe(1);
    s.close();
  });

  it("reads every footer with one plain range request", async () => {
    const s = await open("bands");
    const before = served.length;
    await s.ensure(0, WORLD);
    const reqs = served.slice(before);
    expect(reqs.length).toBe(readSet("bands").shards.filter((x) => x.tier === 0).length);
    for (const r of reqs) expect(r.range).toMatch(/^bytes=\d+-\d+$/);
    s.close();
  });
});
