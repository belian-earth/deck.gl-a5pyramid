// A pyramid set as the renderer reads it: set.json, then each shard's footer
// the first time a view needs the shard, then tiles. Footers are fetched once
// on the main thread and handed to a worker the first time it reads from the
// shard; tiles are read and packed in a pool of workers and kept in a cache
// bounded by bytes.

import {
  boxesMeet,
  checkIndex,
  checkSet,
  setLods,
  tierOf,
  tileBox,
  type Box,
  type PyramidIndex,
  type SetIndex,
  type ShardEntry,
} from "./format.js";
import type { PackedTile, ReadOptions } from "./tile-core.js";
import { readSetTable } from "./set-table.js";
import { boxKey } from "./tiling.js";
import type { Compressors } from "hyparquet";

/** A box meeting every tile. */
export const WORLD: Box = [-180, -90, 180, 90];
import type { WorkerRequest } from "./tile-core.js";

/** Makes one decode worker (worker.ts). */
export type WorkerFactory = () => Worker;

/**
 * The bundled worker, as bundlers expect it (Vite, webpack 5, Rollup and
 * esbuild with their worker support): a module worker from a URL relative
 * to this file. Pass another factory to host the worker elsewhere.
 */
export const defaultWorker: WorkerFactory = () => new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

interface ShardState {
  entry: ShardEntry;
  /** Its name in tile keys: its path, or its tier and cell. */
  key: string;
  url: string;
  footer?: ArrayBuffer;
  index?: PyramidIndex;
  /** Rows and compressed bytes of each tile, from the footer, and the row
   * group of each tile that shares one (format version 4; -1 otherwise). */
  tileRows?: number[];
  tileBytes?: number[];
  tileGroup?: number[];
  opening?: Promise<void>;
  failed?: string;
}

export interface TileRef {
  key: string;
  shard: ShardState;
  tile: number;
  box: Box;
}

/**
 * How tiles are read (the viewer's optimisations of 2026-11-12): one range
 * request a row group read whole, and the PCA sample rate. `merge`: tiles of
 * one row group that a view asks for together share one request a column
 * (default true).
 */
export interface ReadTuning {
  read?: Pick<ReadOptions, "singleRange" | "statsEvery">;
  merge?: boolean;
}

/**
 * How a source is opened. `workers`: decode workers (default: CPUs less
 * two, 2 to 4); `cacheBytes`: decoded tiles kept, least recently used
 * dropped first (default 512 MB); `worker`: how to make a worker (default
 * the bundled one); `refresh`: fetch the set again, for a set whose files
 * are signed URLs that expire (or superseded files that are deleted): a
 * read refused with HTTP 403, 404 or 410 refreshes the set once and is
 * retried; `compressors`: decompressors for a shard table read on the main
 * thread (beyond snappy).
 */
export interface SourceOptions extends ReadTuning {
  workers?: number;
  cacheBytes?: number;
  worker?: WorkerFactory;
  refresh?: () => Promise<SetIndex>;
  compressors?: Compressors;
}

/** A shard's name in tile keys. */
function shardKey(e: ShardEntry): string {
  return e.path ?? `tier=${e.tier}/shard=${e.cell}`;
}

/** Whether a read failed for a URL that has expired or gone. */
function gone(e: unknown): boolean {
  return /HTTP (403|404|410)\b/.test(String((e as Error)?.message ?? e));
}

export interface SourceStats {
  cancelled: number;
  shardsOpen: number;
  footerBytes: number;
  tileBytes: number;
  tilesDecoded: number;
  readMs: number;
  /** Worker time decoding vectors and filling codes textures. */
  decodeMs: number;
  packMs: number;
  /** Worker time spent on tiles' PCA sums. */
  statsMs: number;
  cached: number;
  cachedBytes: number;
  inFlight: number;
}

/** A request for a worker, queued on the main thread until a worker is free. */
interface Job {
  build: (id: number, needsFooter: boolean) => WorkerRequest;
  shard: string;
  resolve: (v: any) => void;
  reject: (e: Error) => void;
  /** Tile layers waiting on it; it is dropped when the last one gives up. */
  waiters: number;
  started: boolean;
  /** Read before any other job (the backdrop level's tiles). */
  urgent: boolean;
  /** Its batch (one per task that submitted jobs) and order of submission. */
  batch: number;
  seq: number;
  /** Jobs of one affinity (the tiles of one row group) go to the worker that
   * ran the last of them when it has room: it holds the row group's page
   * index and dictionaries. */
  affinity?: string;
  /** A tile read that can share one request with other queued tiles of its
   * affinity: `n` its tile, `merge` the request for several. */
  tile?: { n: number; merge: (id: number, needsFooter: boolean, tiles: number[]) => WorkerRequest };
}

/**
 * Workers fed from one queue on the main thread, at most PER_WORKER requests
 * each at a time. The newest batch first (the requests of one task: deck.gl
 * issues a view's tiles together), so the latest view's tiles are read
 * before the ones a camera move passed through; within a batch in the order
 * asked, which is deck.gl's, nearest the centre of the view first. Jobs not
 * yet started are dropped once deck.gl no longer wants them. Urgent jobs
 * (the backdrop level's tiles, shard indexes) go first, and nothing else
 * starts while one is queued or running, so on a slow link they have the
 * bandwidth to themselves.
 */
class Pool {
  static PER_WORKER = 2;
  /** The most tiles of one row group read in one request. */
  static MERGE = 8;
  private workers: { w: Worker; busy: number; shards: Set<string> }[] = [];
  private running = new Map<number, { jobs: Job[]; slot: number }>();
  private pumping = false;
  private queue: Job[] = [];
  private urgent: Job[] = [];
  private urgentRunning = 0;
  private batch = 0;
  private batchOpen = false;
  private seq = 0;
  private next = 1;

  constructor(size: number, make: WorkerFactory) {
    for (let i = 0; i < size; i++) {
      const w = make();
      const slot = i;
      w.onmessage = (ev) => {
        const r = this.running.get(ev.data.id);
        if (!r) return;
        this.running.delete(ev.data.id);
        this.workers[slot].busy--;
        if (r.jobs[0].urgent) this.urgentRunning--;
        if (ev.data.error) for (const j of r.jobs) j.reject(new Error(ev.data.error));
        else if (ev.data.tiles) r.jobs.forEach((j, k) => j.resolve({ ...ev.data, tile: ev.data.tiles[k] }));
        else r.jobs[0].resolve(ev.data);
        this.pump();
      };
      this.workers.push({ w, busy: 0, shards: new Set() });
    }
  }

  get size() {
    return this.workers.length;
  }

  /** Requests queued or running. */
  get inFlight() {
    return this.queue.length + this.urgent.length + this.running.size;
  }

  get queued() {
    return this.queue.length + this.urgent.length;
  }

  submit(build: Job["build"], shard: string, urgent = false, affinity?: string, tile?: Job["tile"]): { job: Job; promise: Promise<any> } {
    let job!: Job;
    const promise = new Promise<any>((resolve, reject) => {
      job = { build, shard, resolve, reject, waiters: 1, started: false, urgent, batch: 0, seq: 0, affinity, tile };
    });
    this.stamp(job);
    (urgent ? this.urgent : this.queue).push(job);
    this.schedule();
    return { job, promise };
  }

  /** Dispatch once the current task has queued its jobs, so a view's tiles
   * of one row group can share a request. */
  private schedule() {
    if (this.pumping) return;
    this.pumping = true;
    queueMicrotask(() => {
      this.pumping = false;
      this.pump();
    });
  }

  /** Move a queued job to the front of its line (the end), or into the urgent one. */
  bump(job: Job, urgent = false) {
    if (job.started) return;
    const from = job.urgent ? this.urgent : this.queue;
    const i = from.indexOf(job);
    if (i < 0) return;
    from.splice(i, 1);
    job.urgent ||= urgent;
    this.stamp(job);
    (job.urgent ? this.urgent : this.queue).push(job);
    this.schedule();
  }

  /** Drop a job not yet started. */
  cancel(job: Job): boolean {
    if (job.started) return false;
    const line = job.urgent ? this.urgent : this.queue;
    const i = line.indexOf(job);
    if (i >= 0) line.splice(i, 1);
    job.reject(new DOMException("tile no longer needed", "AbortError"));
    return true;
  }

  /** Place a job in the current batch, after the jobs already in it. */
  private stamp(job: Job) {
    if (!this.batchOpen) {
      this.batch++;
      this.batchOpen = true;
      setTimeout(() => (this.batchOpen = false), 0);
    }
    job.batch = this.batch;
    job.seq = this.seq++;
  }

  /** Take the next job of `line`: the newest batch, the first asked in it. */
  private take(line: Job[]): Job {
    let best = 0;
    for (let i = 1; i < line.length; i++) {
      const a = line[i], b = line[best];
      if (a.batch > b.batch || (a.batch === b.batch && a.seq < b.seq)) best = i;
    }
    return line.splice(best, 1)[0];
  }

  private affine = new Map<string, number>();

  private pump() {
    while (this.urgent.length || (this.queue.length && !this.urgentRunning)) {
      let slot = -1;
      for (let i = 0; i < this.workers.length; i++) {
        if (this.workers[i].busy < Pool.PER_WORKER && (slot < 0 || this.workers[i].busy < this.workers[slot].busy)) slot = i;
      }
      if (slot < 0) return;
      const job = this.take(this.urgent.length ? this.urgent : this.queue);
      if (job.affinity != null) {
        const pref = this.affine.get(job.affinity);
        if (pref != null && this.workers[pref].busy < Pool.PER_WORKER) slot = pref;
        if (this.affine.size > 4096) this.affine.clear();
        this.affine.set(job.affinity, slot);
      }
      // the other queued tiles of its row group go with it
      const jobs = [job];
      if (job.tile && job.affinity != null) {
        const line = job.urgent ? this.urgent : this.queue;
        for (let i = 0; i < line.length && jobs.length < Pool.MERGE; ) {
          if (line[i].affinity === job.affinity && line[i].tile) jobs.push(line.splice(i, 1)[0]);
          else i++;
        }
      }
      for (const j of jobs) j.started = true;
      if (job.urgent) this.urgentRunning++;
      const w = this.workers[slot];
      const id = this.next++;
      const needsFooter = !w.shards.has(job.shard);
      w.shards.add(job.shard);
      w.busy++;
      this.running.set(id, { jobs, slot });
      w.w.postMessage(jobs.length > 1 ? job.tile!.merge(id, needsFooter, jobs.map((j) => j.tile!.n)) : job.build(id, needsFooter));
    }
  }

  terminate() {
    for (const { w } of this.workers) w.terminate();
    for (const { jobs } of this.running.values()) for (const job of jobs) job.reject(new Error("pyramid closed"));
    for (const job of [...this.queue, ...this.urgent]) job.reject(new Error("pyramid closed"));
    this.running.clear();
    this.queue = [];
  }
}

let sources = 0;

export class PyramidSource {
  /** Unique per source opened: layers drawing it key their state on it. */
  readonly id = ++sources;
  readonly lods: number[];
  readonly columns: string[];
  /** The set's vector column and its elements per cell, or null. */
  readonly vector: { name: string; dim: number } | null;
  /** Bumped whenever a shard opens, so tile selection is redone. */
  version = 0;
  /** Called when a shard opens; the layer redraws. */
  onChange: (() => void) | null = null;
  readonly stats: SourceStats = {
    cancelled: 0, shardsOpen: 0, footerBytes: 0, tileBytes: 0, tilesDecoded: 0, readMs: 0, decodeMs: 0, packMs: 0,
    statsMs: 0, cached: 0, cachedBytes: 0, inFlight: 0,
  };
  /** The set's metadata from its writer (the `a5pyramids` entry), parsed. */
  readonly meta: Record<string, any>;
  /** Whether tiles come with PCA sums: the set's colour is "pca". */
  readonly pca: boolean;

  private shards: ShardState[];
  private byPath = new Map<string, ShardState>();
  private cache = new Map<string, Promise<PackedTile>>();
  private jobs = new Map<string, Job>();
  private sizes = new Map<string, number>();
  private pool: Pool;

  private readonly refreshSet?: () => Promise<SetIndex>;
  private refreshing: Promise<boolean> | null = null;
  private readonly read: ReadTuning["read"];
  private readonly merge: boolean;

  private constructor(readonly base: string, public set: SetIndex, workers: number, readonly cacheBytes: number, worker: WorkerFactory,
    opts: SourceOptions = {}) {
    this.read = opts.read ?? {};
    this.merge = opts.merge ?? true;
    this.refreshSet = opts.refresh;
    this.lods = setLods(set);
    // scalar columns by name; the vector column (an embedding), if any, apart
    this.columns = set.columns.filter((c) => c.dim == null).map((c) => c.name);
    const v = set.columns.find((c) => c.dim != null);
    this.vector = v ? { name: v.name, dim: v.dim! } : null;
    try {
      this.meta = JSON.parse(set.metadata?.a5pyramids ?? "{}");
    } catch {
      this.meta = {};
    }
    this.pca = this.meta.colour === "pca";
    // shards listed as empty have no file
    this.shards = set.shards
      .filter((e) => e.path != null || e.url != null)
      .map((entry) => ({ entry, key: shardKey(entry), url: entry.url ?? new URL(entry.path!, base).href }));
    for (const s of this.shards) this.byPath.set(s.key, s);
    this.pool = new Pool(workers, worker);
  }

  /**
   * Open the set at `base`: a directory URL (ending in `/`) holding
   * `set.json` or `set.parquet`, or the URL of either file.
   */
  static async open(base: string, opts: SourceOptions = {}): Promise<PyramidSource> {
    // relative to the page, where there is one
    const page = globalThis.location?.href;
    const file = /\.(json|parquet)$/.test(new URL(base, page).pathname);
    const dir = new URL(file ? "." : base.endsWith("/") ? base : base + "/", new URL(base, page)).href;
    let set: SetIndex;
    if (file && base.endsWith(".parquet")) set = await readSetTable(new URL(base, page).href, opts.compressors);
    else {
      const json = file ? new URL(base, page).href : new URL("set.json", dir).href;
      const res = await fetch(json);
      // a directory without set.json has set.parquet: a store answers 403 or
      // 404, a development server its page
      let parsed: SetIndex | null = null;
      if (res.ok) {
        const text = await res.text();
        try {
          parsed = JSON.parse(text);
        } catch (e) {
          if (file) throw new Error(`${json}: ${e}`);
        }
      } else if (file || (res.status !== 404 && res.status !== 403)) throw new Error(`${json}: HTTP ${res.status}`);
      set = parsed ?? (await readSetTable(new URL("set.parquet", dir).href, opts.compressors));
    }
    return PyramidSource.fromSet(set, { ...opts, base: dir });
  }

  /**
   * A source over a set given whole, as a service hands one out: its fields
   * and its shards, each with a `url` (such as a signed URL) or a `path`
   * relative to `base`.
   */
  static fromSet(set: SetIndex, opts: SourceOptions & { base?: string } = {}): PyramidSource {
    const workers = opts.workers ?? Math.max(2, Math.min(4, (globalThis.navigator?.hardwareConcurrency || 4) - 2));
    const base = new URL(opts.base ?? ".", globalThis.location?.href ?? "http://localhost/").href;
    return new PyramidSource(base, checkSet(set), workers, opts.cacheBytes ?? 512 * 1024 * 1024, opts.worker ?? defaultWorker, opts);
  }

  /**
   * Fetch the set again (with `refresh`) and take its URLs: a shard whose
   * file changed (another build) is reopened and its tiles dropped. Once at a
   * time; resolves false without `refresh`.
   */
  private renew(): Promise<boolean> {
    if (!this.refreshSet) return Promise.resolve(false);
    this.refreshing ??= this.refreshSet()
      .then((fresh) => {
        const set = checkSet(fresh);
        const known = new Map(this.shards.map((s) => [`${s.entry.tier}/${s.entry.cell}`, s]));
        for (const e of set.shards) {
          const s = known.get(`${e.tier}/${e.cell}`);
          if (!s || (e.path == null && e.url == null)) continue;
          const url = e.url ?? new URL(e.path!, this.base).href;
          const same = e.path === s.entry.path && e.bytes === s.entry.bytes && e.footer_bytes === s.entry.footer_bytes;
          s.url = url;
          if (!same) {
            s.entry = e;
            s.index = s.footer = s.tileRows = s.tileBytes = s.tileGroup = s.opening = undefined;
            s.failed = undefined;
            for (const k of [...this.cache.keys()]) if (k.startsWith(s.key + "#")) this.forget(k);
          } else s.failed = undefined;
        }
        this.set = set;
        this.version++;
        this.onChange?.();
        return true;
      })
      .catch((e) => {
        console.error(`[deck.gl-a5pyramid] refreshing the set: ${e}`);
        return false;
      })
      .finally(() => (this.refreshing = null));
    return this.refreshing;
  }

  get workers() {
    return this.pool.size;
  }

  /** Requests sent to the workers and not yet answered. */
  get inFlight() {
    return this.pool.inFlight + this.footersOpening;
  }
  /** Footers being read (before their index reaches a worker). */
  private footersOpening = 0;

  close() {
    this.pool.terminate();
    this.cache.clear();
  }

  tierOf(lod: number) {
    return tierOf(this.set, lod);
  }

  /** Shards of `tier` meeting `box`, open or not. */
  shardsIn(tier: number, box: Box): ShardState[] {
    return this.shards.filter((s) => s.entry.tier === tier && s.entry.bbox != null && boxesMeet(s.entry.bbox, box));
  }

  /** Start opening the shards of `tier` meeting `box`; resolves when all have. */
  ensure(tier: number, box: Box): Promise<void> {
    return Promise.all(this.shardsIn(tier, box).map((s) => this.openShard(s))).then(() => undefined);
  }

  /** Whether every shard of `tier` meeting `box` has opened (or failed). */
  opened(tier: number, box: Box): boolean {
    return this.shardsIn(tier, box).every((s) => s.index || s.failed);
  }

  /** Whether `lod` can be selected for `box`: its shards there have opened. */
  ready(lod: number, box: Box): boolean {
    return this.opened(this.tierOf(lod), box);
  }

  private openShard(s: ShardState, retry = true): Promise<void> {
    if (s.index || s.failed) return Promise.resolve();
    if (s.opening) return s.opening;
    const { bytes, footer_bytes } = s.entry;
    s.opening = (async () => {
      this.footersOpening++;
      try {
        const res = await fetch(s.url, { headers: { Range: `bytes=${bytes - footer_bytes}-${bytes - 1}` } });
        // an expired or superseded file: the set once more, then this shard again
        if ((res.status === 403 || res.status === 404 || res.status === 410) && retry && (await this.renew())) {
          s.opening = undefined;
          await this.openShard(s, false);
          return;
        }
        if (res.status !== 206 && res.status !== 200) throw new Error(`HTTP ${res.status}`);
        let footer = await res.arrayBuffer();
        if (res.status === 200 && footer.byteLength === bytes) footer = footer.slice(bytes - footer_bytes);
        this.stats.footerBytes += footer.byteLength;
        s.footer = footer;
        const out = await this.pool.submit(
          (id) => ({ id, type: "index", url: s.url, bytes, footer: footer.slice(0) }),
          s.url,
          true, // a shard's index decides which tiles exist: before any tile
        ).promise;
        const index = checkIndex(JSON.parse(out.kv.a5pyramid), out.kv.a5_index, s.key);
        const n = index.tile_bbox.length / 4;
        if (out.tileRows.length !== n) throw new Error(`the index has ${n} tiles, the footer ${out.tileRows.length}`);
        s.tileRows = out.tileRows;
        s.tileBytes = out.tileBytes;
        s.tileGroup = out.tileGroup;
        s.index = index;
        this.stats.shardsOpen++;
        this.version++;
        this.onChange?.();
      } catch (e) {
        s.failed = String((e as Error)?.message ?? e);
        console.error(`[deck.gl-a5pyramid] ${s.key}: ${s.failed}`);
      } finally {
        this.footersOpening--;
      }
    })();
    return s.opening;
  }

  /** The tiles at `lod` of open shards whose boxes meet `box`. */
  tilesIn(lod: number, box: Box): TileRef[] {
    const tier = this.tierOf(lod);
    const out: TileRef[] = [];
    for (const s of this.shardsIn(tier, box)) {
      const idx = s.index;
      if (!idx) continue;
      const level = idx.levels.find((l) => l.lod === lod);
      if (!level) continue;
      for (let i = level.tiles[0]; i < level.tiles[1]; i++) {
        const b = tileBox(idx, i);
        if (boxesMeet(b, box)) out.push({ key: `${s.key}#${i}`, shard: s, tile: i, box: b });
      }
    }
    return out;
  }

  /**
   * The base level: the finest level of the coarsest tier whose tiles over
   * the whole set are within `budget`, or null when even its coarsest is
   * not. Opens every shard of that tier (a few small footers).
   */
  async baseLevel(budget: { cells: number; bytes: number }): Promise<number | null> {
    const fits = (rows: number, bytes: number) => rows > 0 && rows <= budget.cells && bytes <= budget.bytes;
    // a tier of one level costs what its shards hold, from the set alone
    const whole = (t: number) => {
      let rows = 0, bytes = 0;
      for (const s of this.shards) if (s.entry.tier === t) (rows += s.entry.rows), (bytes += s.entry.bytes);
      return fits(rows, bytes);
    };
    const tiers = this.set.tiers;
    if (!tiers.length) return null;
    let best: number | null = null;
    if (tiers[0].lods[0] === tiers[0].lods[1]) {
      if (!whole(0)) return null;
      best = tiers[0].lods[0];
    } else {
      await this.ensure(0, WORLD);
      for (let lod = tiers[0].lods[1]; lod >= tiers[0].lods[0] && best == null; lod--) {
        const c = this.costIn(lod, WORLD);
        if (c && fits(c.rows, c.bytes)) best = lod;
      }
      if (best !== tiers[0].lods[1]) return best;
    }
    // finer tiers of one level each (a store's), while they fit
    for (let t = 1; t < tiers.length && tiers[t].lods[0] === tiers[t].lods[1] && whole(t); t++) best = tiers[t].lods[0];
    return best;
  }

  /**
   * What reading `lod` for `box` costs: the rows and compressed bytes of its
   * tiles meeting the box; null until every shard of its tier there has
   * opened.
   */
  costIn(lod: number, box: Box): { rows: number; bytes: number; tiles: number } | null {
    const key = `${lod}|${boxKey(box)}|${this.version}`;
    const known = this.costs.get(key);
    if (known) return known;
    if (!this.opened(this.tierOf(lod), box)) return null;
    let rows = 0, bytes = 0, tiles = 0;
    for (const t of this.tilesIn(lod, box)) {
      rows += t.shard.tileRows?.[t.tile] ?? 0;
      bytes += t.shard.tileBytes?.[t.tile] ?? 0;
      tiles++;
    }
    if (this.costs.size > 512) this.costs.clear();
    const c = { rows, bytes, tiles };
    this.costs.set(key, c);
    return c;
  }
  private costs = new Map<string, { rows: number; bytes: number; tiles: number }>();

  tileBox(key: string): Box | null {
    const k = key.lastIndexOf("#");
    const s = this.byPath.get(key.slice(0, k));
    return s?.index ? tileBox(s.index, Number(key.slice(k + 1))) : null;
  }

  /**
   * The packed tile `key` (`<path>#<tile>`), decoded once and cached. A tile
   * still queued when every caller's `signal` has aborted is dropped; an
   * `urgent` one is read before any other.
   */
  tile(key: string, lod: number, signal?: AbortSignal, urgent = false): Promise<PackedTile> {
    const hit = this.cache.get(key);
    if (hit) {
      // most recently used last
      this.cache.delete(key);
      this.cache.set(key, hit);
      const job = this.jobs.get(key);
      if (job) {
        job.waiters++;
        this.pool.bump(job, urgent);
        this.watch(key, job, signal);
      }
      return hit;
    }
    const k = key.lastIndexOf("#");
    const i = Number(key.slice(k + 1));
    const s = this.byPath.get(key.slice(0, k));
    if (!s?.index || !s.footer) return Promise.reject(new Error(`shard not open: ${key.slice(0, k)}`));
    const { job, promise: first } = this.submitTile(s, i, lod, urgent);
    // a read refused for an expired or superseded file: the set once more
    const promise = first.catch(async (e) => {
      if (!gone(e) || !(await this.renew()) || !s.index || !s.footer) throw e;
      return this.submitTile(s, i, lod, urgent).promise;
    });
    const p = promise.then((out) => {
      const t = out.tile as PackedTile;
      this.stats.tilesDecoded++;
      this.stats.tileBytes += t.fetched;
      this.stats.readMs += t.readMs;
      this.stats.decodeMs += t.decodeMs ?? 0;
      this.stats.packMs += t.packMs;
      this.stats.statsMs += t.statsMs;
      if (this.jobs.get(key) === job) this.jobs.delete(key);
      this.remember(key, t.byteLength);
      return t;
    });
    p.catch(() => {
      if (this.jobs.get(key) === job) {
        this.jobs.delete(key);
        this.forget(key);
      }
    });
    this.cache.set(key, p);
    this.jobs.set(key, job);
    this.watch(key, job, signal);
    return p;
  }

  /** Queue the read of tile `i` of `s`; tiles of one row group can share a request. */
  private submitTile(s: ShardState, i: number, lod: number, urgent: boolean) {
    const footer = s.footer!;
    const read: ReadOptions = { ...this.read };
    const group = s.tileGroup?.[i] ?? -1;
    const url = s.url;
    const request = (id: number, needsFooter: boolean, tiles: number[]): WorkerRequest => ({
      id, type: "decode", url, bytes: s.entry.bytes, footer: needsFooter ? footer.slice(0) : undefined,
      tile: tiles[0], tiles: tiles.length > 1 ? tiles : undefined, lod, columns: this.columns, stats: this.pca, vector: this.vector, read,
    });
    return this.pool.submit(
      (id, needsFooter) => request(id, needsFooter, [i]),
      url,
      urgent,
      group >= 0 ? `${url}#${group}` : undefined,
      group >= 0 && this.merge ? { n: i, merge: request } : undefined,
    );
  }

  private watch(key: string, job: Job, signal?: AbortSignal) {
    if (!signal) return;
    const off = () => {
      job.waiters--;
      if (job.waiters <= 0 && this.pool.cancel(job)) this.stats.cancelled++;
    };
    if (signal.aborted) off();
    else signal.addEventListener("abort", off, { once: true });
  }

  private remember(key: string, bytes: number) {
    this.sizes.set(key, bytes);
    this.stats.cachedBytes += bytes;
    for (const k of this.cache.keys()) {
      if (this.stats.cachedBytes <= this.cacheBytes) break;
      if (k === key || !this.sizes.has(k)) continue;
      this.forget(k);
    }
    this.stats.cached = this.sizes.size;
  }

  private forget(key: string) {
    this.cache.delete(key);
    const b = this.sizes.get(key);
    if (b != null) {
      this.stats.cachedBytes -= b;
      this.sizes.delete(key);
    }
    this.stats.cached = this.sizes.size;
  }
}
