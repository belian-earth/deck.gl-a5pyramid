// The browser tour over local store sets, one layout or read setting against
// another. TOUR_DATA: a directory holding the sets; TOUR_SETS: the sets to
// compare, comma separated, each a directory of TOUR_DATA (holding set.json
// or set.parquet); TOUR_STEPS: a JSON file of the steps (name, center, zoom,
// lod), else the Tiisi tour below. Skipped unless TOUR_BENCH is set:
//
//   TOUR_BENCH=1 TOUR_DATA=... TOUR_SETS=a,b TOUR_STEPS=steps.json npx vitest run test/tour-bench.test.ts
//
// The sets of 2026-11-12 (Tiisi's nine store partitions, ramet branch
// store-v4, dev/bench/v4) were deleted after the row-group decision; rebuild
// them to run this. The decision's figures (MB, requests, worker s): 16,384-row
// groups 92.3, 111, 7.0; 131,072-row groups with 16,384-row pages read as
// tiles 90.9, 258, 6.8; the same read whole 213.2, 43, 15.1.
//
// The views are the browser tour's, at its levels, in a 1036 × 663 px map,
// one source a variant so tiles cached by earlier steps are reused as they
// would be. Worker time is in-process, one core.
import { describe, expect, it } from "vitest";
import { existsSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { ZSTDDecoder } from "zstddec";
import { PyramidSource, type ReadTuning } from "../src/source.js";
import { setCompressors, type PackedTile } from "../src/tile-core.js";
import { metresPerPixel } from "../src/tiling.js";
import type { Box } from "../src/format.js";
import { localWorker } from "./helpers.js";

const DATA = process.env.TOUR_DATA ?? "";
const ORIGIN = "http://tour.test/";
const W = 1036, H = 663;
type Step = { name: string; center: [number, number]; zoom: number; lod: number };
const TIISI: Step[] = [
  { name: "Tiisi", center: [-1.0, 10.1], zoom: 7.6, lod: 13 },
  { name: "one partition", center: [-0.979, 10.04], zoom: 10, lod: 15 },
  { name: "leaves", center: [-0.979, 10.04], zoom: 13.5, lod: 18 },
  { name: "pan east", center: [-0.91, 10.04], zoom: 13.5, lod: 18 },
  { name: "pan north", center: [-0.91, 10.09], zoom: 13.5, lod: 18 },
  { name: "another partition", center: [-1.72, 9.966], zoom: 10, lod: 15 },
  { name: "its leaves", center: [-1.72, 9.966], zoom: 13, lod: 18 },
  { name: "Tiisi again", center: [-1.0, 10.1], zoom: 7.6, lod: 13 },
];
const STEPS: Step[] = process.env.TOUR_STEPS ? JSON.parse(readFileSync(process.env.TOUR_STEPS, "utf8")) : TIISI;

const served = { bytes: 0, requests: 0 };
const files = new Map<string, { fd: number; size: number }>();
function installLocalFetch() {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.origin + "/" !== ORIGIN) return real(input, init);
    const path = DATA + "/" + decodeURIComponent(url.pathname.slice(1));
    if (!files.has(path) && (!existsSync(path) || !statSync(path).isFile())) return new Response("not found", { status: 404 });
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

async function runTour(set: string, opts: ReadTuning) {
  const source = await PyramidSource.open(`${ORIGIN}${set}/`, { worker: localWorker, workers: 4, ...opts });
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
      pca: Math.round(d("statsMs")),
      cpu: Math.round(d("readMs") + d("decodeMs") + d("packMs") + d("statsMs")),
      wall: Math.round(wall),
    });
  }
  source.close();
  return rows;
}

describe.skipIf(!process.env.TOUR_BENCH)("tour benchmark", () => {
  it("measures the tour under each variant", async () => {
    installLocalFetch();
    const zd = new ZSTDDecoder();
    await zd.init();
    setCompressors({ ZSTD: (i: Uint8Array, n: number) => zd.decode(i, n) });
    // TOUR_SETS: the sets to compare, comma separated (default: tiisi, one
    // tile a row group, and tiisi-v4, format version 4)
    const sets = (process.env.TOUR_SETS ?? "tiisi,tiisi-v4").split(",");
    const variants: [string, string, ReadTuning][] = sets.flatMap((set): [string, string, ReadTuning][] => [
      [`${set} (defaults: merged reads, one range a whole row group, PCA 1/16)`, set, {}],
      [`${set}, tiles read one at a time`, set, { merge: false }],
    ]);
    const out: string[] = [];
    // a first pass warms the JIT and the page cache; the second is reported
    for (const [, set, opts] of variants) await runTour(set, opts);
    for (const [name, set, opts] of variants) {
      const rows = await runTour(set, opts);
      const total = (k: string) => rows.reduce((a, r) => a + (r[k] as number), 0);
      out.push(`\n== ${name}\n` + rows.map((r) => Object.entries(r).map(([k, v]) => `${k} ${v}`).join(", ")).join("\n") +
        `\nTOTAL MB ${total("MB").toFixed(1)}, requests ${total("requests")}, cells ${total("cells")}, cpu ${total("cpu")} ms ` +
        `(read ${total("read")}, decode ${total("decode")}, pack ${total("pack")}, pca ${total("pca")})`);
    }
    const { writeFileSync } = await import("node:fs");
    writeFileSync(process.env.TOUR_OUT ?? "/dev/stdout", out.join("\n") + "\n");
    expect(out.length).toBe(variants.length);
  }, 1_800_000);
});
