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
import { boxKey } from "./tiling.js";

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
  url: string;
  footer?: ArrayBuffer;
  index?: PyramidIndex;
  /** Rows and compressed bytes of each tile (row group), from the footer. */
  tileRows?: number[];
  tileBytes?: number[];
  opening?: Promise<void>;
  failed?: string;
  /** Pieces of the tiles: their rows, first piece of each tile, and boxes
   * (from the index, or probed tile by tile). */
  pieceRows?: number;
  pieceStart?: number[];
  pieceBox?: Float32Array;
  probes?: Map<number, Float32Array>;
  probing?: Set<number>;
}

export interface TileRef {
  key: string;
  shard: ShardState;
  tile: number;
  /** The piece of the tile, when tiles are read in pieces. */
  piece?: number;
  box: Box;
}

/**
 * How tiles are cut and read (the viewer's optimisations of 2026-11-12).
 * `pieces`: "auto" reads the pieces a set's index declares (pages cut at
 * `piece_rows`, a box each) and whole tiles otherwise; "probe" also cuts
 * tiles without them into pieces of `pieceRows`, boxes probed from each
 * tile's cells; "off" reads whole tiles. `read`: one range request a tile,
 * and the PCA sample rate.
 */
export interface PieceOptions {
  pieces?: "auto" | "probe" | "off";
  pieceRows?: number;
  read?: Pick<ReadOptions, "singleRange" | "statsEvery">;
}

/** Floats of base64 (little-endian float32). */
function base64Floats(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
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
  /** Probes of tiles for their pieces' boxes: bytes and worker time. */
  probes: number;
  probeBytes: number;
  probeMs: number;
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
  /** Jobs of one affinity (the pieces of a tile) go to the worker that ran
   * the last of them when it has room: it holds their row group. */
  affinity?: string;
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
  private workers: { w: Worker; busy: number; shards: Set<string> }[] = [];
  private running = new Map<number, { job: Job; slot: number }>();
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
        if (r.job.urgent) this.urgentRunning--;
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
    return this.queue.length + this.urgent.length + this.running.size;
  }

  get queued() {
    return this.queue.length + this.urgent.length;
  }

  submit(build: Job["build"], shard: string, urgent = false, affinity?: string): { job: Job; promise: Promise<any> } {
    let job!: Job;
    const promise = new Promise<any>((resolve, reject) => {
      job = { build, shard, resolve, reject, waiters: 1, started: false, urgent, batch: 0, seq: 0, affinity };
    });
    this.stamp(job);
    (urgent ? this.urgent : this.queue).push(job);
    this.pump();
    return { job, promise };
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
    this.pump();
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
      job.started = true;
      if (job.urgent) this.urgentRunning++;
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
    probes: 0, probeBytes: 0, probeMs: 0, statsMs: 0, cached: 0, cachedBytes: 0, inFlight: 0,
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

  private constructor(readonly base: string, readonly set: SetIndex, workers: number, readonly cacheBytes: number, worker: WorkerFactory,
    readonly pieces: Required<Omit<PieceOptions, "read">> & { read: PieceOptions["read"] } = { pieces: "auto", pieceRows: 2048, read: {} }) {
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
    this.shards = set.shards.map((entry) => ({ entry, url: new URL(entry.path, base).href }));
    for (const s of this.shards) this.byPath.set(s.entry.path, s);
    this.pool = new Pool(workers, worker);
  }

  /**
   * Open the set at `base` (a URL ending in `/`, holding set.json).
   * `workers`: decode workers (default: CPUs less two, 2 to 4);
   * `cacheBytes`: decoded tiles kept, least recently used dropped first
   * (default 512 MB); `worker`: how to make a worker (default the bundled one).
   */
  static async open(base: string, opts: { workers?: number; cacheBytes?: number; worker?: WorkerFactory } & PieceOptions = {}): Promise<PyramidSource> {
    // relative to the page, where there is one
    const url = new URL(base.endsWith("/") ? base : base + "/", globalThis.location?.href).href;
    const res = await fetch(new URL("set.json", url));
    if (!res.ok) throw new Error(`${url}set.json: HTTP ${res.status}`);
    const set = checkSet(await res.json());
    const workers = opts.workers ?? Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2));
    return new PyramidSource(url, set, workers, opts.cacheBytes ?? 512 * 1024 * 1024, opts.worker ?? defaultWorker, {
      pieces: opts.pieces ?? "auto", pieceRows: opts.pieceRows ?? 2048, read: opts.read ?? {},
    });
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

  /** Whether `lod` can be selected for `box`: its shards there have opened
   * and, when tiles are probed for pieces, its tiles there have been. */
  ready(lod: number, box: Box): boolean {
    if (!this.opened(this.tierOf(lod), box)) return false;
    if (this.pieces.pieces !== "probe") return true;
    // tiles not yet probed are probed now; the level waits for them
    let ready = true;
    for (const s of this.shardsIn(this.tierOf(lod), box)) {
      if (!s.index || s.pieceBox) continue;
      const level = s.index.levels.find((l) => l.lod === lod);
      if (!level) continue;
      for (let i = level.tiles[0]; i < level.tiles[1]; i++) {
        if (!s.probes?.has(i) && boxesMeet(tileBox(s.index, i), box)) {
          this.probe(s, i, lod);
          ready = false;
        }
      }
    }
    return ready;
  }

  /** Probe tile `i` of `s` for its pieces' boxes (once). */
  private probe(s: ShardState, i: number, lod: number) {
    s.probes ??= new Map();
    s.probing ??= new Set();
    if (s.probes.has(i) || s.probing.has(i) || !s.footer) return;
    s.probing.add(i);
    const footer = s.footer;
    const pieceRows = this.pieces.pieceRows;
    this.pool.submit(
      (id, needsFooter) => ({ id, type: "probe", url: s.url, bytes: s.entry.bytes, footer: needsFooter ? footer.slice(0) : undefined, tile: i, lod, pieceRows }),
      s.url, false, `${s.url}#${i}`,
    ).promise.then((out) => {
      s.probes!.set(i, out.probe.boxes as Float32Array);
      this.stats.probes++;
      this.stats.probeBytes += out.probe.fetched;
      this.stats.probeMs += out.probe.ms;
      this.version++;
      this.onChange?.();
    }, (e) => console.error(`[deck.gl-a5pyramid] probe ${s.entry.path}#${i}: ${e}`)).finally(() => s.probing!.delete(i));
  }

  /** The pieces' boxes of tile `i` of `s`, and its first piece's position in
   * them; null when the tile is read whole (or not yet probed). */
  private piecesOf(s: ShardState, i: number): { boxes: Float32Array; first: number; count: number } | null {
    if (this.pieces.pieces === "off") return null;
    const rows = s.tileRows?.[i] ?? 0;
    if (s.pieceBox && s.pieceStart && s.pieceRows) return { boxes: s.pieceBox, first: s.pieceStart[i], count: s.pieceStart[i + 1] - s.pieceStart[i] };
    if (this.pieces.pieces === "probe") {
      const b = s.probes?.get(i);
      return b ? { boxes: b, first: 0, count: Math.ceil(rows / this.pieces.pieceRows) } : null;
    }
    return null;
  }

  /** Rows of a piece's tile cut, and its rows. */
  private pieceRowsOf(s: ShardState): number {
    return s.pieceBox ? s.pieceRows! : this.pieces.pieceRows;
  }

  private openShard(s: ShardState): Promise<void> {
    if (s.index || s.failed) return Promise.resolve();
    if (s.opening) return s.opening;
    const { bytes, footer_bytes } = s.entry;
    s.opening = (async () => {
      this.footersOpening++;
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
          true, // a shard's index decides which tiles exist: before any tile
        ).promise;
        s.index = checkIndex(JSON.parse(out.kv.a5pyramid), out.kv.a5_index, s.entry.path);
        s.tileRows = out.tileRows;
        s.tileBytes = out.tileBytes;
        const ix = s.index as PyramidIndex & { piece_rows?: number; piece_bbox_f32?: string };
        if (this.pieces.pieces !== "off" && ix.piece_rows && ix.piece_bbox_f32) {
          s.pieceRows = ix.piece_rows;
          s.pieceBox = base64Floats(ix.piece_bbox_f32);
          s.pieceStart = [0];
          for (const r of s.tileRows!) s.pieceStart.push(s.pieceStart[s.pieceStart.length - 1] + Math.ceil(r / ix.piece_rows));
        }
        this.stats.shardsOpen++;
        this.version++;
        this.onChange?.();
      } catch (e) {
        s.failed = String((e as Error)?.message ?? e);
        console.error(`[deck.gl-a5pyramid] ${s.entry.path}: ${s.failed}`);
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
        if (!boxesMeet(b, box)) continue;
        const ps = this.piecesOf(s, i);
        if (ps) {
          for (let k = 0; k < ps.count; k++) {
            const o = (ps.first + k) * 4;
            const pb: Box = [ps.boxes[o], ps.boxes[o + 1], ps.boxes[o + 2], ps.boxes[o + 3]];
            if (boxesMeet(pb, box)) out.push({ key: `${s.entry.path}#${i}.${k}`, shard: s, tile: i, piece: k, box: pb });
          }
        } else if (this.pieces.pieces === "probe" && !s.pieceBox) {
          this.probe(s, i, lod);
        } else out.push({ key: `${s.entry.path}#${i}`, shard: s, tile: i, box: b });
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
    await this.ensure(0, WORLD);
    const tier = this.set.tiers[0];
    if (!tier) return null;
    for (let lod = tier.lods[1]; lod >= tier.lods[0]; lod--) {
      const c = this.costIn(lod, WORLD);
      if (c && c.rows > 0 && c.rows <= budget.cells && c.bytes <= budget.bytes) return lod;
    }
    return null;
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
      const tr = t.shard.tileRows?.[t.tile] ?? 0;
      const tb = t.shard.tileBytes?.[t.tile] ?? 0;
      if (t.piece != null) {
        const pr = this.pieceRowsOf(t.shard);
        const r = Math.min(pr, tr - t.piece * pr);
        rows += r;
        bytes += tr ? (tb * r) / tr : 0;
      } else {
        rows += tr;
        bytes += tb;
      }
      tiles++;
    }
    if (this.costs.size > 512) this.costs.clear();
    const c = { rows, bytes, tiles };
    this.costs.set(key, c);
    return c;
  }
  private costs = new Map<string, { rows: number; bytes: number; tiles: number }>();

  tileBox(key: string): Box | null {
    const [path, ref] = key.split("#");
    const s = this.byPath.get(path);
    if (!s?.index) return null;
    const [i, k] = ref.split(".").map(Number);
    if (k == null || Number.isNaN(k)) return tileBox(s.index, i);
    const ps = this.piecesOf(s, i);
    if (!ps) return tileBox(s.index, i);
    const o = (ps.first + k) * 4;
    return [ps.boxes[o], ps.boxes[o + 1], ps.boxes[o + 2], ps.boxes[o + 3]];
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
    const [path, ref] = key.split("#");
    const [i, k] = ref.split(".").map(Number);
    const s = this.byPath.get(path);
    if (!s?.index || !s.footer) return Promise.reject(new Error(`shard not open: ${path}`));
    const footer = s.footer;
    const piece = k == null || Number.isNaN(k) ? undefined : k;
    const read: ReadOptions = {
      ...this.pieces.read,
      ...(piece != null ? { piece, pieceRows: this.pieceRowsOf(s), pages: !!s.pieceBox } : {}),
    };
    const { job, promise } = this.pool.submit(
      (id, needsFooter) => ({
        id, type: "decode", url: s.url, bytes: s.entry.bytes, footer: needsFooter ? footer.slice(0) : undefined,
        tile: i, lod, columns: this.columns, stats: this.pca, vector: this.vector, read,
      }),
      s.url,
      urgent,
      piece != null && !s.pieceBox ? `${s.url}#${i}` : undefined,
    );
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
