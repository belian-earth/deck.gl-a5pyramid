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
} from "./format";
import type { PackedTile } from "./tile-core";
import type { WorkerRequest } from "./worker";
import PyramidWorker from "./worker?worker";

interface ShardState {
  entry: ShardEntry;
  url: string;
  footer?: ArrayBuffer;
  index?: PyramidIndex;
  opening?: Promise<void>;
  failed?: string;
}

export interface TileRef {
  key: string;
  shard: ShardState;
  tile: number;
  box: Box;
}

export interface SourceStats {
  cancelled: number;
  shardsOpen: number;
  footerBytes: number;
  tileBytes: number;
  tilesDecoded: number;
  readMs: number;
  packMs: number;
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
}

/**
 * Workers fed from one queue on the main thread, at most PER_WORKER requests
 * each at a time, newest first: the latest view's tiles are read before the
 * ones a camera move passed through, and those not yet started are dropped
 * once deck.gl no longer wants them.
 */
class Pool {
  static PER_WORKER = 2;
  private workers: { w: Worker; busy: number; shards: Set<string> }[] = [];
  private running = new Map<number, { job: Job; slot: number }>();
  private queue: Job[] = [];
  private next = 1;

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      const w = new PyramidWorker();
      const slot = i;
      w.onmessage = (ev) => {
        const r = this.running.get(ev.data.id);
        if (!r) return;
        this.running.delete(ev.data.id);
        this.workers[slot].busy--;
        if (ev.data.error) r.job.reject(new Error(ev.data.error));
        else r.job.resolve(ev.data);
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
    return this.queue.length + this.running.size;
  }

  get queued() {
    return this.queue.length;
  }

  submit(build: Job["build"], shard: string): { job: Job; promise: Promise<any> } {
    let job!: Job;
    const promise = new Promise<any>((resolve, reject) => {
      job = { build, shard, resolve, reject, waiters: 1, started: false };
    });
    this.queue.push(job);
    this.pump();
    return { job, promise };
  }

  /** Move a queued job to the front of the line (the end of the queue). */
  bump(job: Job) {
    if (job.started) return;
    const i = this.queue.indexOf(job);
    if (i >= 0 && i !== this.queue.length - 1) {
      this.queue.splice(i, 1);
      this.queue.push(job);
    }
  }

  /** Drop a job not yet started. */
  cancel(job: Job): boolean {
    if (job.started) return false;
    const i = this.queue.indexOf(job);
    if (i >= 0) this.queue.splice(i, 1);
    job.reject(new DOMException("tile no longer needed", "AbortError"));
    return true;
  }

  private pump() {
    while (this.queue.length) {
      let slot = -1;
      for (let i = 0; i < this.workers.length; i++) {
        if (this.workers[i].busy < Pool.PER_WORKER && (slot < 0 || this.workers[i].busy < this.workers[slot].busy)) slot = i;
      }
      if (slot < 0) return;
      const job = this.queue.pop()!;
      job.started = true;
      const w = this.workers[slot];
      const id = this.next++;
      const needsFooter = !w.shards.has(job.shard);
      w.shards.add(job.shard);
      w.busy++;
      this.running.set(id, { job, slot });
      w.w.postMessage(job.build(id, needsFooter));
    }
  }

  terminate() {
    for (const { w } of this.workers) w.terminate();
    for (const { job } of this.running.values()) job.reject(new Error("pyramid closed"));
    for (const job of this.queue) job.reject(new Error("pyramid closed"));
    this.running.clear();
    this.queue = [];
  }
}

export class PyramidSource {
  readonly lods: number[];
  readonly columns: string[];
  /** Bumped whenever a shard opens, so tile selection is redone. */
  version = 0;
  /** Called when a shard opens; the layer redraws. */
  onChange: (() => void) | null = null;
  readonly stats: SourceStats = {
    cancelled: 0, shardsOpen: 0, footerBytes: 0, tileBytes: 0, tilesDecoded: 0, readMs: 0, packMs: 0, cached: 0, cachedBytes: 0, inFlight: 0,
  };

  private shards: ShardState[];
  private byPath = new Map<string, ShardState>();
  private cache = new Map<string, Promise<PackedTile>>();
  private jobs = new Map<string, Job>();
  private sizes = new Map<string, number>();
  private pool: Pool;

  private constructor(readonly base: string, readonly set: SetIndex, workers: number, readonly cacheBytes: number) {
    this.lods = setLods(set);
    this.columns = set.columns.map((c) => c.name);
    this.shards = set.shards.map((entry) => ({ entry, url: new URL(entry.path, base).href }));
    for (const s of this.shards) this.byPath.set(s.entry.path, s);
    this.pool = new Pool(workers);
  }

  /** Open the set at `base` (a URL ending in `/`, holding set.json). */
  static async open(base: string, opts: { workers?: number; cacheBytes?: number } = {}): Promise<PyramidSource> {
    const url = new URL(base.endsWith("/") ? base : base + "/", location.href).href;
    const res = await fetch(new URL("set.json", url));
    if (!res.ok) throw new Error(`${url}set.json: HTTP ${res.status}`);
    const set = checkSet(await res.json());
    const workers = opts.workers ?? Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2));
    return new PyramidSource(url, set, workers, opts.cacheBytes ?? 512 * 1024 * 1024);
  }

  get workers() {
    return this.pool.size;
  }

  /** Requests sent to the workers and not yet answered. */
  get inFlight() {
    return this.pool.inFlight;
  }

  close() {
    this.pool.terminate();
    this.cache.clear();
  }

  tierOf(lod: number) {
    return tierOf(this.set, lod);
  }

  /** Shards of `tier` meeting `box`, open or not. */
  shardsIn(tier: number, box: Box): ShardState[] {
    return this.shards.filter((s) => s.entry.tier === tier && boxesMeet(s.entry.bbox, box));
  }

  /** Start opening the shards of `tier` meeting `box`; resolves when all have. */
  ensure(tier: number, box: Box): Promise<void> {
    return Promise.all(this.shardsIn(tier, box).map((s) => this.openShard(s))).then(() => undefined);
  }

  /** Whether every shard of `tier` meeting `box` has opened (or failed). */
  opened(tier: number, box: Box): boolean {
    return this.shardsIn(tier, box).every((s) => s.index || s.failed);
  }

  private openShard(s: ShardState): Promise<void> {
    if (s.index || s.failed) return Promise.resolve();
    if (s.opening) return s.opening;
    const { bytes, footer_bytes } = s.entry;
    s.opening = (async () => {
      try {
        const res = await fetch(s.url, { headers: { Range: `bytes=${bytes - footer_bytes}-${bytes - 1}` } });
        if (res.status !== 206 && res.status !== 200) throw new Error(`HTTP ${res.status}`);
        let footer = await res.arrayBuffer();
        if (res.status === 200 && footer.byteLength === bytes) footer = footer.slice(bytes - footer_bytes);
        this.stats.footerBytes += footer.byteLength;
        s.footer = footer;
        const out = await this.pool.submit(
          (id) => ({ id, type: "index", url: s.url, bytes, footer: footer.slice(0) }),
          s.url,
        ).promise;
        s.index = checkIndex(JSON.parse(out.kv.a5pyramid), out.kv.a5_index, s.entry.path);
        this.stats.shardsOpen++;
        this.version++;
        this.onChange?.();
      } catch (e) {
        s.failed = String((e as Error)?.message ?? e);
        console.error(`[a5-pyramids] ${s.entry.path}: ${s.failed}`);
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
        if (boxesMeet(b, box)) out.push({ key: `${s.entry.path}#${i}`, shard: s, tile: i, box: b });
      }
    }
    return out;
  }

  tileBox(key: string): Box | null {
    const [path, i] = key.split("#");
    const s = this.byPath.get(path);
    return s?.index ? tileBox(s.index, Number(i)) : null;
  }

  /**
   * The packed tile `key` (`<path>#<tile>`), decoded once and cached. A tile
   * still queued when every caller's `signal` has aborted is dropped.
   */
  tile(key: string, lod: number, signal?: AbortSignal): Promise<PackedTile> {
    const hit = this.cache.get(key);
    if (hit) {
      // most recently used last
      this.cache.delete(key);
      this.cache.set(key, hit);
      const job = this.jobs.get(key);
      if (job) {
        job.waiters++;
        this.pool.bump(job);
        this.watch(key, job, signal);
      }
      return hit;
    }
    const [path, i] = key.split("#");
    const s = this.byPath.get(path);
    if (!s?.index || !s.footer) return Promise.reject(new Error(`shard not open: ${path}`));
    const footer = s.footer;
    const { job, promise } = this.pool.submit(
      (id, needsFooter) => ({
        id, type: "decode", url: s.url, bytes: s.entry.bytes, footer: needsFooter ? footer.slice(0) : undefined,
        tile: Number(i), lod, columns: this.columns,
      }),
      s.url,
    );
    const p = promise.then((out) => {
      const t = out.tile as PackedTile;
      this.stats.tilesDecoded++;
      this.stats.tileBytes += t.fetched;
      this.stats.readMs += t.readMs;
      this.stats.packMs += t.packMs;
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
