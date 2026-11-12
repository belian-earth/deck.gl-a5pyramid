import { beforeAll, describe, expect, it } from "vitest";
import { PyramidSource, WORLD } from "../src/source.js";
import { readSetTable } from "../src/set-table.js";
import type { SetIndex } from "../src/format.js";
import { ORIGIN, installFetch, localWorker, served } from "./helpers.js";

// The store in miniature (tools/fixture-store-v4.R): one level a file and a
// tier, levels 15 to 17 built per partition with partial rows and owner,
// list vectors, n_pixels, and set.parquet with an empty partition.
beforeAll(installFetch);

const SET = `${ORIGIN}store-v4/`;
const open = (opts = {}) => PyramidSource.open(SET, { worker: localWorker, workers: 2, ...opts });

async function readLevel(s: PyramidSource, lod: number) {
  await s.ensure(s.tierOf(lod), WORLD);
  const refs = s.tilesIn(lod, WORLD);
  const tiles = await Promise.all(refs.map((r) => s.tile(r.key, lod)));
  return { refs, tiles };
}

describe("a store's set", () => {
  it("opens from its shard table, skipping the empty partition", async () => {
    const s = await open();
    expect(s.set.format_version).toBe(4);
    expect(s.set.tiers.length).toBe(10);
    expect(s.set.shards.some((e) => e.path == null && e.rows === 0 && e.bbox == null)).toBe(true);
    expect(s.lods).toEqual([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    expect(s.vector).toEqual({ name: "mean_embedding", dim: 64 });
    expect(s.columns).toEqual([]);
    expect(s.pca).toBe(true);
    const { tiles } = await readLevel(s, 18);
    expect(tiles.reduce((a, t) => a + t.n, 0)).toBe(1011);
    s.close();
  });

  it("draws every partial row: a split cell's rows count its leaves between them", async () => {
    const s = await open();
    const leaves = (await readLevel(s, 18)).tiles;
    const level = (await readLevel(s, 16)).tiles;
    const rows = level.reduce((a, t) => a + t.n, 0);
    const cells = new Map<bigint, number>();
    for (const t of level) for (let i = 0; i < t.n; i++) cells.set(t.cells[i], (cells.get(t.cells[i]) ?? 0) + t.ncell[i]);
    // some cells have rows in two partitions, and every leaf is counted once
    expect(rows).toBeGreaterThan(cells.size);
    expect([...cells.values()].reduce((a, b) => a + b, 0)).toBe(leaves.reduce((a, t) => a + t.n, 0));
    // the weights are not read: a tile holds its vector and no value column
    expect(level[0].values.length).toBe(0);
    expect(level[0].vector!.length).toBe(level[0].n * 64);
    s.close();
  });

  it("reads tiles of one row group together, with fewer requests and the same rows", async () => {
    const counted = async (merge: boolean) => {
      const s = await open({ merge });
      await s.ensure(s.tierOf(18), WORLD);
      const before = served.length;
      const { tiles } = await readLevel(s, 18);
      const n = served.length - before;
      s.close();
      return { n, cells: tiles.flatMap((t) => Array.from(t.cells)) };
    };
    const one = await counted(false);
    const many = await counted(true);
    expect(many.cells).toEqual(one.cells);
    expect(many.n).toBeLessThan(one.n / 2);
  });

  it("picks the base level from the set's rows, across tiers of one level", async () => {
    const s = await open();
    const before = served.length;
    expect(await s.baseLevel({ cells: 100, bytes: 1e9 })).toBe(16);
    expect(await s.baseLevel({ cells: 2000, bytes: 1e9 })).toBe(18);
    expect(await s.baseLevel({ cells: 1, bytes: 1e9 })).toBe(9);
    expect(await s.baseLevel({ cells: 0, bytes: 1e9 })).toBe(null);
    // without opening a footer
    expect(served.length).toBe(before);
    s.close();
  });
});

describe("a set given whole, as a service signs it", () => {
  it("reads files by their URLs and refreshes them once they expire", async () => {
    const table = await readSetTable(`${SET}set.parquet`);
    let signs = 0;
    const signed = (sig: string): SetIndex => ({
      ...table,
      shards: table.shards.map((e) => ({ ...e, url: e.path ? `${SET}${e.path}?sig=${sig}` : undefined, path: null })),
    });
    // a signature the stub refuses, as CloudFront refuses an expired one
    const real = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) =>
      String(input).includes("sig=old") ? new Response("expired", { status: 403 }) : real(input, init)) as typeof fetch;
    try {
      const s = PyramidSource.fromSet(signed("old"), {
        worker: localWorker, workers: 2,
        refresh: async () => (signs++, signed("new")),
      });
      const { tiles } = await readLevel(s, 17);
      expect(signs).toBe(1);
      expect(tiles.reduce((a, t) => a + t.n, 0)).toBe(297);
      // once fresh, reads go through without another refresh
      await readLevel(s, 18);
      expect(signs).toBe(1);
      s.close();
    } finally {
      globalThis.fetch = real;
    }
  });
});
