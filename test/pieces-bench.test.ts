// The viewer's optimisations of 2026-11-12, measured one against another on
// a local store v4 set (PIECES_DATA: a directory holding tiisi/ and
// tiisi-pages/ year=2021 sets; ZSTDDEC: zstddec's ESM build). Skipped unless
// PIECES_BENCH is set:
//
//   PIECES_BENCH=1 PIECES_DATA=... ZSTDDEC=... npx vitest run test/pieces-bench.test.ts
//
// The views are the browser tour's, at its levels, in a 1036 × 663 px map,
// one source a variant so tiles cached by earlier steps are reused as they
// would be. Worker time is in-process, one core.
import { describe, expect, it } from "vitest";
import { openSync, readSync, statSync } from "node:fs";
import { PyramidSource, type PieceOptions } from "../src/source.js";
import { setCompressors, type PackedTile } from "../src/tile-core.js";
import { metresPerPixel } from "../src/tiling.js";
import type { Box } from "../src/format.js";
import { localWorker } from "./helpers.js";

const DATA = process.env.PIECES_DATA ?? "";
const ORIGIN = "http://pieces.test/";
const W = 1036, H = 663;
const STEPS: { name: string; center: [number, number]; zoom: number; lod: number }[] = [
  { name: "Tiisi", center: [-1.0, 10.1], zoom: 7.6, lod: 13 },
  { name: "one partition", center: [-0.979, 10.04], zoom: 10, lod: 15 },
  { name: "leaves", center: [-0.979, 10.04], zoom: 13.5, lod: 18 },
  { name: "pan east", center: [-0.91, 10.04], zoom: 13.5, lod: 18 },
  { name: "pan north", center: [-0.91, 10.09], zoom: 13.5, lod: 18 },
  { name: "another partition", center: [-1.72, 9.966], zoom: 10, lod: 15 },
  { name: "its leaves", center: [-1.72, 9.966], zoom: 13, lod: 18 },
  { name: "Tiisi again", center: [-1.0, 10.1], zoom: 7.6, lod: 13 },
];

const served = { bytes: 0, requests: 0 };
const files = new Map<string, { fd: number; size: number }>();
function installLocalFetch() {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.origin + "/" !== ORIGIN) return real(input, init);
    const path = DATA + "/" + decodeURIComponent(url.pathname.slice(1));
    let f = files.get(path);
    if (!f) files.set(path, (f = { fd: openSync(path, "r"), size: statSync(path).size }));
    const { fd, size } = f;
    const m = /^bytes=(\d+)-(\d+)$/.exec(new Headers(init?.headers).get("range") ?? "");
    const a = m ? Number(m[1]) : 0, b = m ? Math.min(Number(m[2]), size - 1) : size - 1;
    const buf = Buffer.alloc(b - a + 1);
    readSync(fd, buf, 0, buf.length, a);
    served.bytes += buf.length;
    served.requests++;
    return new Response(new Uint8Array(buf), { status: m ? 206 : 200 });
  }) as typeof fetch;
}

/** The view's box, widened by 10% a side as viewportBox widens it. */
function viewBox(center: [number, number], zoom: number): Box {
  const mpp = metresPerPixel(zoom, center[1]);
  const dLon = (W / 2) * mpp / (111320 * Math.cos((center[1] * Math.PI) / 180));
  const dLat = (H / 2) * mpp / 110574;
  return [center[0] - 1.2 * dLon, center[1] - 1.2 * dLat, center[0] + 1.2 * dLon, center[1] + 1.2 * dLat];
}

async function until(f: () => boolean) {
  for (let i = 0; i < 20000 && !f(); i++) await new Promise((r) => setTimeout(r, 2));
}

async function runTour(set: string, opts: PieceOptions) {
  const source = await PyramidSource.open(`${ORIGIN}${set}/year=2021/`, { worker: localWorker, workers: 4, ...opts });
  const rows: Record<string, number | string>[] = [];
  const seen = new Set<string>();
  for (const st of STEPS) {
    const box = viewBox(st.center, st.zoom);
    const b0 = served.bytes, r0 = served.requests;
    const s0 = { ...source.stats };
    await source.ensure(source.tierOf(st.lod), box);
    await until(() => source.ready(st.lod, box));
    const refs = source.tilesIn(st.lod, box);
    const fresh = refs.filter((r) => !seen.has(r.key));
    const t0 = performance.now();
    const tiles: PackedTile[] = await Promise.all(refs.map((r) => source.tile(r.key, st.lod)));
    const wall = performance.now() - t0;
    for (const r of refs) seen.add(r.key);
    const d = (k: keyof typeof s0) => (source.stats[k] as number) - (s0[k] as number);
    rows.push({
      step: st.name, lod: st.lod, tiles: refs.length, new: fresh.length,
      cells: tiles.reduce((a, t) => a + t.n, 0),
      MB: +((served.bytes - b0) / 1e6).toFixed(2), requests: served.requests - r0,
      read: Math.round(d("readMs")), decode: Math.round(d("decodeMs")), pack: Math.round(d("packMs")),
      pca: Math.round(d("statsMs")), probe: Math.round(d("probeMs")),
      cpu: Math.round(d("readMs") + d("decodeMs") + d("packMs") + d("statsMs") + d("probeMs")),
      wall: Math.round(wall),
    });
  }
  source.close();
  return rows;
}

describe.skipIf(!process.env.PIECES_BENCH)("pieces benchmark", () => {
  it("measures the tour under each variant", async () => {
    installLocalFetch();
    const { ZSTDDecoder } = await import(process.env.ZSTDDEC!);
    const zd = new ZSTDDecoder();
    await zd.init();
    setCompressors({ ZSTD: (i: Uint8Array, n: number) => zd.decode(i, n) });
    // PIECES_VARIANTS=rg: the row-group decision of 2026-11-12 (ramet wants
    // large row groups; the viewer pieces of 16,384 rows)
    const variants: [string, string, PieceOptions][] = process.env.PIECES_VARIANTS === "rg" ? [
      ["16,384-row groups (defaults: one range, PCA 1/16)", "tiisi", { pieces: "off" }],
      ["131,072-row groups, pages of 16,384 read as pieces", "tiisi-rg", { pieces: "auto" }],
      ["131,072-row groups read whole", "tiisi-rg", { pieces: "off" }],
    ] : [
      ["baseline", "tiisi", { pieces: "off", read: { singleRange: false, statsEvery: 4 } }],
      ["3: PCA 1/16", "tiisi", { pieces: "off", read: { singleRange: false, statsEvery: 16 } }],
      ["4: one range a tile", "tiisi", { pieces: "off", read: { singleRange: true, statsEvery: 4 } }],
      ["2: probed pieces", "tiisi", { pieces: "probe", pieceRows: 2048, read: { singleRange: false, statsEvery: 4 } }],
      ["2-4: probed pieces, one range, PCA 1/16", "tiisi", { pieces: "probe", pieceRows: 2048, read: { singleRange: true, statsEvery: 16 } }],
      ["3 + 4", "tiisi", { pieces: "off", read: { singleRange: true, statsEvery: 16 } }],
      ["baseline on the paged files", "tiisi-pages", { pieces: "off", read: { singleRange: false, statsEvery: 4 } }],
      ["1: index pieces, page reads", "tiisi-pages", { pieces: "auto", read: { statsEvery: 4 } }],
      ["1 + 3: and PCA 1/16", "tiisi-pages", { pieces: "auto", read: { statsEvery: 16 } }],
    ];
    const out: string[] = [];
    // a first pass warms the JIT and the page cache; the second is reported
    for (const [, set, opts] of variants) await runTour(set, opts);
    for (const [name, set, opts] of variants) {
      const rows = await runTour(set, opts);
      const total = (k: string) => rows.reduce((a, r) => a + (r[k] as number), 0);
      out.push(`\n== ${name}\n` + rows.map((r) => Object.entries(r).map(([k, v]) => `${k} ${v}`).join(", ")).join("\n") +
        `\nTOTAL MB ${total("MB").toFixed(1)}, requests ${total("requests")}, cells ${total("cells")}, cpu ${total("cpu")} ms ` +
        `(read ${total("read")}, decode ${total("decode")}, pack ${total("pack")}, pca ${total("pca")}, probe ${total("probe")})`);
    }
    const { writeFileSync } = await import("node:fs");
    writeFileSync(process.env.PIECES_OUT ?? "/dev/stdout", out.join("\n") + "\n");
    expect(out.length).toBe(variants.length);
  }, 1_800_000);
});
