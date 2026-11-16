// A tour over a remote set given whole (a service's signed URLs, as
// belianapi hands a job's embeddings out), through the source as the layer
// drives it: for each view the footers of the backdrop's tier and the
// level's, the level within the budget, then the backdrop's tiles (urgent)
// and the level's. Skipped unless TOUR_REMOTE names a JSON file holding the
// set (`{set: ...}` or the set itself); TOUR_REGION a JSON file of outer
// rings (the region drawn); TOUR_STEPS the steps (name, center, zoom; the
// level follows from the zoom at 2 px a cell), else test/tour-para.json;
// TOUR_VARIANTS the source options to compare, as JSON objects separated by
// semicolons (default: the defaults alone); TOUR_OUT a file for the report.
// TOUR_MODEL replays the tour through a model of a CDN instead of the
// network: `latency=<s>,bandwidth=<bytes/s>,cache=<dir>`; every range read
// is kept under `cache` (fetched once, from the real URL) and a request
// takes the latency plus its bytes at the bandwidth shared by the requests
// in flight. Run the tour once to fill the cache (TOUR_FILL=1 runs each
// variant twice and reports the second).
//
//   TOUR_REMOTE=set.json TOUR_REGION=rings.json npx vitest run test/tour-remote.test.ts
//
// Workers are threads (worker_threads over the built package: `npm run
// build` first), or in-process with TOUR_INPROCESS=1 (decoding then serial).
import { describe, expect, it } from "vitest";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { ZSTDDecoder } from "zstddec";
import { PyramidSource, WORLD, type SourceOptions } from "../src/source.js";
import { setCompressors, type PackedTile } from "../src/tile-core.js";
import { a5Resolution, metresPerPixel, pickLod } from "../src/tiling.js";
import { BASE_BUDGET, DEFAULT_BUDGET, chooseLevel } from "../src/budget.js";
import { BACKDROP_STEP } from "../src/layer.js";
import type { Box, SetIndex } from "../src/format.js";
import type { Ring } from "../src/region.js";
import { localWorker, nodeWorker } from "./helpers.js";

const W = 1036, H = 663;
type Step = { name: string; center: [number, number]; zoom: number };

/** Requests to the set's files: count, bytes, how many ran at once, their times. */
const net = { requests: 0, bytes: 0, inFlight: 0, maxInFlight: 0, ms: [] as number[] };

/** The CDN model (TOUR_MODEL), or null for the network. */
const MODEL = (() => {
  const m = process.env.TOUR_MODEL;
  if (!m) return null;
  const kv = Object.fromEntries(m.split(",").map((x) => x.split("=") as [string, string]));
  return { latency: Number(kv.latency ?? 0.1), bandwidth: Number(kv.bandwidth ?? 12e6), cache: kv.cache ?? "tour-cache" };
})();

/** Ranges kept on disk by file: `<cache>/<hash of the URL's path>/<lo>-<hi>`. */
const kept = new Map<string, { lo: number; hi: number; file: string }[]>();
function keptDir(url: string) {
  const dir = `${MODEL!.cache}/${createHash("sha1").update(new URL(url).pathname).digest("hex").slice(0, 16)}`;
  if (!kept.has(dir)) {
    mkdirSync(dir, { recursive: true });
    kept.set(dir, readdirSync(dir).map((f) => {
      const [lo, hi] = f.split("-").map(Number);
      return { lo, hi, file: `${dir}/${f}` };
    }));
  }
  return { dir, pieces: kept.get(dir)! };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function installCountingFetch() {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const t0 = performance.now();
    net.requests++;
    net.inFlight++;
    net.maxInFlight = Math.max(net.maxInFlight, net.inFlight);
    try {
      const range = /^bytes=(\d+)-(\d+)$/.exec(new Headers(init?.headers).get("range") ?? "");
      if (MODEL && range) {
        const lo = Number(range[1]), hi = Number(range[2]) + 1;
        const { dir, pieces } = keptDir(url);
        let have = pieces.find((p) => p.lo <= lo && hi <= p.hi);
        if (!have) {
          const res = await real(input, init);
          if (res.status !== 206) throw new Error(`${url}: HTTP ${res.status} filling the cache`);
          const buf = Buffer.from(await res.arrayBuffer());
          have = { lo, hi: lo + buf.length, file: `${dir}/${lo}-${lo + buf.length}` };
          writeFileSync(have.file, buf);
          pieces.push(have);
        }
        const buf = readFileSync(have.file).subarray(lo - have.lo, hi - have.lo);
        // the model: latency, then the bytes at the bandwidth shared by the
        // requests in flight when this one starts
        await sleep(1000 * (MODEL.latency + buf.length / (MODEL.bandwidth / Math.max(1, net.inFlight))));
        net.bytes += buf.length;
        net.ms.push(performance.now() - t0);
        return new Response(new Uint8Array(buf), { status: 206, headers: { "content-range": `bytes ${lo}-${hi - 1}/*` } });
      }
      const res = await real(input, init);
      const buf = await res.arrayBuffer();
      net.bytes += buf.byteLength;
      net.ms.push(performance.now() - t0);
      return new Response(buf, { status: res.status, headers: res.headers });
    } finally {
      net.inFlight--;
    }
  }) as typeof fetch;
}

/** The view's box, widened by 10% a side as viewportBox widens it. */
function viewBox(center: [number, number], zoom: number): Box {
  const mpp = metresPerPixel(zoom, center[1]);
  const dLon = ((W / 2) * mpp) / (111320 * Math.cos((center[1] * Math.PI) / 180));
  const dLat = ((H / 2) * mpp) / 110574;
  return [center[0] - 1.2 * dLon, center[1] - 1.2 * dLat, center[0] + 1.2 * dLon, center[1] + 1.2 * dLat];
}

async function until(f: () => boolean, ms = 120_000) {
  const t0 = performance.now();
  while (!f() && performance.now() - t0 < ms) await new Promise((r) => setTimeout(r, 2));
}

const ms = (t: number) => Math.round(t);

/** One view, as the layer drives the source. */
async function view(source: PyramidSource, st: Step) {
  const box = viewBox(st.center, st.zoom);
  const vp = { zoom: st.zoom, latitude: st.center[1] };
  const t0 = performance.now();
  const n0 = { ...net };
  const s0 = { ...source.stats };
  const target = pickLod(a5Resolution(vp, 2), source.lods);
  // the footers of the backdrop's tier and the level's at once, as the
  // layer asks for them, then the level within the budget, each further
  // candidate's tier opened as the layer asks for it
  void source.ensure(source.tierOf(pickLod(target - BACKDROP_STEP, source.lods)), box);
  void source.ensure(source.tierOf(pickLod(target - 1, source.lods)), box);
  let choice = chooseLevel(target, source.lods, (lod) => (void source.ensure(source.tierOf(lod), box), source.costIn(lod, box)), DEFAULT_BUDGET);
  while (!choice.cost) {
    await until(() => source.ready(choice.lod, box));
    choice = chooseLevel(target, source.lods, (lod) => (void source.ensure(source.tierOf(lod), box), source.costIn(lod, box)), DEFAULT_BUDGET);
  }
  const lod = choice.lod;
  const back = pickLod(lod - BACKDROP_STEP, source.lods);
  const backdrop = back < lod ? back : null;
  if (backdrop != null) await source.ensure(source.tierOf(backdrop), box);
  await source.ensure(source.tierOf(lod), box);
  const tFooters = performance.now() - t0;
  const bt = backdrop != null ? source.tilesIn(backdrop, box) : [];
  const lt = source.tilesIn(lod, box);
  let firstBack = NaN, firstTile = NaN;
  const backTiles = Promise.all(bt.map((r) => source.tile(r.key, backdrop!, undefined, true).then((t) => ((firstBack ||= performance.now() - t0), t))));
  const tiles = Promise.all(lt.map((r) => source.tile(r.key, lod).then((t) => ((firstTile ||= performance.now() - t0), t))));
  const backs: PackedTile[] = await backTiles;
  const tBack = performance.now() - t0;
  const packed: PackedTile[] = await tiles;
  const tAll = performance.now() - t0;
  const d = (k: keyof typeof s0) => (source.stats[k] as number) - (s0[k] as number);
  return {
    step: st.name, target, lod, capped: choice.capped ?? "", backdrop: backdrop ?? "",
    footers: ms(tFooters), firstBack: ms(firstBack), backAll: ms(tBack), firstTile: ms(firstTile), all: ms(tAll),
    tiles: lt.length, backTiles: bt.length, cells: packed.reduce((a, t) => a + t.n, 0) + backs.reduce((a, t) => a + t.n, 0),
    MB: +((net.bytes - n0.bytes) / 1e6).toFixed(1), requests: net.requests - n0.requests,
    footerMs: ms(d("footerMs")), indexMs: ms(d("indexMs")),
    cpu: ms(d("readMs") + d("decodeMs") + d("packMs") + d("statsMs")),
  };
}

async function runTour(set: SetIndex, region: Ring[] | null, steps: Step[], opts: SourceOptions) {
  net.requests = net.bytes = net.maxInFlight = 0;
  net.ms = [];
  const t0 = performance.now();
  const source = PyramidSource.fromSet(set, { worker: process.env.TOUR_INPROCESS ? localWorker : nodeWorker, workers: 4, pca: true, cacheBytes: 128 * 1024 * 1024, ...(region ? { region } : {}), ...opts });
  const base = await source.baseLevel(BASE_BUDGET);
  const rows: Record<string, number | string>[] = [{ step: "open", base: base ?? "", all: ms(performance.now() - t0) }];
  // the base level's tiles, read in the background as the base layer reads them
  let baseDone = NaN, baseCells = 0;
  if (base != null) {
    void source.ensure(source.tierOf(base), WORLD).then(() => Promise.all(source.tilesIn(base, WORLD).map((r) => source.tile(r.key, base)))).then((ts) => {
      baseDone = performance.now() - t0;
      baseCells = ts.reduce((a, t) => a + t.n, 0);
    });
  }
  for (const st of steps) rows.push(await view(source, st));
  rows.push({ step: "base", lod: base ?? "", all: ms(baseDone), cells: baseCells });
  const total = performance.now() - t0;
  const sorted = [...net.ms].sort((a, b) => a - b);
  const q = (p: number) => ms(sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] ?? 0);
  rows.push({ step: "TOTAL", all: ms(total), MB: +(net.bytes / 1e6).toFixed(1), requests: net.requests, maxInFlight: net.maxInFlight, reqMedianMs: q(0.5), reqP90Ms: q(0.9), skipped: source.stats.tilesSkipped, dropped: source.stats.rowsDropped });
  source.close();
  return rows;
}

describe.skipIf(!process.env.TOUR_REMOTE)("remote tour", () => {
  it("drives the source through the tour under each variant", async () => {
    installCountingFetch();
    const zd = new ZSTDDecoder();
    await zd.init();
    setCompressors({ ZSTD: (i: Uint8Array, n: number) => zd.decode(i, n) });
    const given = JSON.parse(readFileSync(process.env.TOUR_REMOTE!, "utf8"));
    const set: SetIndex = given.set ?? given;
    const region: Ring[] | null = process.env.TOUR_REGION ? JSON.parse(readFileSync(process.env.TOUR_REGION, "utf8")) : null;
    const steps: Step[] = JSON.parse(readFileSync(process.env.TOUR_STEPS ?? new URL("./tour-para.json", import.meta.url), "utf8"));
    const variants: SourceOptions[] = (process.env.TOUR_VARIANTS ?? "{}").split(";").map((s) => JSON.parse(s));
    const out: string[] = [];
    for (const opts of variants) {
      if (process.env.TOUR_FILL) await runTour(set, region, steps, opts);
      const rows = await runTour(set, region, steps, opts);
      out.push(`\n== ${JSON.stringify(opts)}\n` + rows.map((r) => Object.entries(r).map(([k, v]) => `${k} ${v}`).join(", ")).join("\n"));
    }
    writeFileSync(process.env.TOUR_OUT ?? "/dev/stdout", out.join("\n") + "\n");
    expect(out.length).toBe(variants.length);
  }, 1_800_000);
});
