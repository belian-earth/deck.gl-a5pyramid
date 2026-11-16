// A pyramid set as the renderer reads it: set.json, then each shard's footer
// the first time a view needs the shard, then tiles. Footers are fetched once
// on the main thread and handed to a worker the first time it reads from the
// shard, whose index lists the byte ranges each tile needs; a tile's bytes
// are fetched on the main thread, many at a time, and decoded and packed in
// a pool of workers; packed tiles are kept in a cache bounded by bytes.

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
import type { Covering, PackedTile, Piece, ReadOptions } from "./tile-core.js";
import { readSetTable } from "./set-table.js";
import { checkRings, ringBox, type Ring } from "./region.js";
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
  /** Each tile's byte ranges (tile-core flatRanges), from the footer. */
  tileRanges?: Float64Array;
  rangeStarts?: Uint32Array;
  /** With a region: whether each tile can hold one of its cells (from the
   * worker that read the footer). */
  tileMeets?: Uint8Array;
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
 * one row group that a view asks for together share one request a column.
 * Default: off for a set whose files are http(s) URLs, on otherwise (local
 * files). Over a CDN merging cut the requests in flight and slowed a tour by
 * about 60% with several connections, and gained nothing over one HTTP/2
 * connection (2026-11-13); on local files it saves requests at no cost.
 * `connections`: tile fetches in flight at once (default 16; urgent reads
 * may add URGENT_CONNECTIONS more). `block`: a range smaller than this is
 * fetched as the whole blocks of this size holding it, kept for the file's
 * other tiles (a row group's cell pages and dictionary pages lie together,
 * a few hundred bytes each); default 64 KB, 0 for none.
 */
export interface ReadTuning {
  read?: Pick<ReadOptions, "singleRange" | "statsEvery">;
  merge?: boolean;
  connections?: number;
  block?: number;
}

/**
 * How a source is opened. `workers`: decode workers (default: CPUs less
 * two, 2 to 4); `cacheBytes`: decoded tiles kept, least recently used
 * dropped first (default 512 MB); `worker`: how to make a worker (default
 * the bundled one); `pool`: a pool of workers shared between sources
 * (createWorkerPool), which `close` leaves running, so a new source over
 * the same region starts with the workers, their shards' indexes and the
 * region's coverings in place; `refresh`: fetch the set again, for a set
 * whose files are signed URLs that expire (or superseded files that are
 * deleted): a read refused with HTTP 403, 404 or 410 refreshes the set once
 * and is retried; `compressors`: decompressors for a shard table read on
 * the main thread (beyond snappy); `pca`: tiles come with PCA sums, for
 * ScreenPca, whatever the set's metadata says (default: its `colour` is
 * "pca"), as for a set an app colours by screen PCA but whose writer left
 * no hint.
 */
export interface SourceOptions extends ReadTuning {
  workers?: number;
  cacheBytes?: number;
  worker?: WorkerFactory;
  pool?: WorkerPool;
  refresh?: () => Promise<SetIndex>;
  compressors?: Compressors;
  pca?: boolean;
  /**
   * Draw only the cells of a region: polygons as their outer rings (holes
   * filled by the caller), `[longitude, latitude]` points, not crossing the
   * antimeridian. Each level's cells are those of a5-js `polygonToCells`
   * (containment "overlapping") of each ring, unioned, so a cell is drawn
   * whole or not at all, and PCA sums are taken over the region's cells
   * alone. Built in the workers, a level when first read, and shared: a
   * covering one worker computes reaches the others, and sources over the
   * same rings (another year of a store) reuse them. Shards and tiles
   * that cannot hold one of its cells are not read: a shard by its box, a
   * tile by its cell id bounds in the footer (statistics or column index)
   * against the level's covering, else by its box against the rings'.
   */
  region?: Ring[];
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
  /** Bytes fetched for tiles (blocks and ranges, as requested). */
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
  /** Tiles of opened shards that cannot hold a cell of the region, so are never read. */
  tilesSkipped: number;
  /** Rows read and dropped as outside the region. */
  rowsDropped: number;
  /** Time fetching footers, and worker time opening them (parsing, the region's coverings). */
  footerMs: number;
  indexMs: number;
  /** Requests made for tiles. */
  tileRequests: number;
}

// ---- the region's coverings, shared -------------------------------------

/** A key of a region's rings: sources over the same rings share one region. */
export function regionId(rings: Ring[]): number {
  const s = JSON.stringify(rings);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
  return h || 1;
}

/** Coverings by region, as workers report them: given to the workers that lack them. */
const coverings = new Map<number, Map<string, BigUint64Array>>();
const COVERINGS_REGIONS = 8;

function rememberCoverings(list: Covering[]): void {
  for (const c of list) {
    let m = coverings.get(c.region);
    if (!m) {
      coverings.delete(c.region);
      coverings.set(c.region, (m = new Map()));
      while (coverings.size > COVERINGS_REGIONS) coverings.delete(coverings.keys().next().value!);
    }
    if (!m.has(c.key)) m.set(c.key, c.ranges);
  }
}

// ---- the pool -----------------------------------------------------------

/** A request for a worker, queued on the main thread: fetched, then decoded. */
interface Job {
  /** The source it belongs to. */
  owner: number;
  build: (id: number, needsFooter: boolean, pieces: Piece[]) => WorkerRequest;
  shard: string;
  url: string;
  /** The file's byte ranges the job reads; null for an index job, which reads nothing. */
  ranges: [number, number][] | null;
  /** Where the file's data ends (its footer starts): a block is clipped to it. */
  dataEnd: number;
  region: { id: number; rings: Ring[] } | null;
  resolve: (v: any) => void;
  reject: (e: Error) => void;
  /** Tile layers waiting on it; it is dropped when the last one gives up. */
  waiters: number;
  state: "queued" | "fetching" | "ready" | "running" | "done";
  /** Read before any other job (the backdrop level's tiles, shard indexes). */
  urgent: boolean;
  /** Its batch (one per task that submitted jobs) and order of submission. */
  batch: number;
  seq: number;
  /** Jobs of one affinity (the tiles of one row group) go to the worker that
   * ran the last of them when it has room: it holds the row group's page
   * index. */
  affinity?: string;
  /** A tile read that can share one request with other queued tiles of its
   * affinity: `n` its tile, `merge` the request for several. */
  tile?: { n: number; merge: (id: number, needsFooter: boolean, tiles: number[], pieces: Piece[]) => WorkerRequest };
  group?: Group;
}

/** Jobs fetched and decoded together: one, or merged tiles of a row group. */
interface Group {
  jobs: Job[];
  pieces: Piece[];
  /** Whether each piece is the group's own (transferred) or a kept block (copied). */
  owned: boolean[];
  aborts: AbortController[];
  cancelled: boolean;
}

/** A pool of decode workers, shared between sources (createWorkerPool). */
export interface WorkerPool {
  readonly size: number;
  /** Requests queued or running, every source's. */
  readonly inFlight: number;
  terminate(): void;
}

/** Urgent reads in flight beyond `connections`, so they never wait on the others. */
export const URGENT_CONNECTIONS = 8;

/**
 * Workers fed from one queue on the main thread, in two stages. A job's
 * bytes are fetched first, up to `connections` jobs at once (urgent ones
 * beyond that), small ranges as kept blocks; a job whose bytes are in goes
 * to a worker with a free slot (PER_WORKER each), with the bytes
 * transferred. Both stages take the urgent jobs first (the backdrop level's
 * tiles, shard indexes), then the newest batch (the requests of one task:
 * deck.gl issues a view's tiles together), so the latest view's tiles are
 * read before the ones a camera move passed through; within a batch in the
 * order asked, which is deck.gl's, nearest the centre of the view first.
 * A job not yet started is dropped once deck.gl no longer wants it, and a
 * fetch is aborted once no job sharing it is wanted.
 */
class Pool implements WorkerPool {
  static PER_WORKER = 2;
  /** The most tiles of one row group read in one request. */
  static MERGE = 8;
  /** Kept blocks at most. */
  static BLOCKS = 512;
  private workers: { w: Worker; busy: number; shards: Set<string>; regions: Set<number>; coverings: Set<string> }[] = [];
  private running = new Map<number, { group: Group; slot: number }>();
  private pumping = false;
  private queue: Job[] = [];
  private urgent: Job[] = [];
  private ready: Group[] = [];
  private fetching = 0;
  private fetchingUrgent = 0;
  private batch = 0;
  private batchOpen = false;
  private seq = 0;
  private next = 1;
  private blocks = new Map<string, Promise<ArrayBuffer>>();
  private owners = new Map<number, { count: number; fetched: (bytes: number, requests: number) => void }>();
  readonly connections: number;
  readonly block: number;

  constructor(size: number, make: WorkerFactory, connections = 16, block = 64 * 1024) {
    this.connections = Math.max(1, connections);
    this.block = Math.max(0, block);
    for (let i = 0; i < size; i++) {
      const w = make();
      const slot = i;
      w.onmessage = (ev) => {
        const r = this.running.get(ev.data.id);
        if (!r) return;
        this.running.delete(ev.data.id);
        this.workers[slot].busy--;
        if (ev.data.coverings) {
          rememberCoverings(ev.data.coverings);
          for (const c of ev.data.coverings as Covering[]) this.workers[slot].coverings.add(`${c.region}|${c.key}`);
        }
        const { jobs } = r.group;
        if (ev.data.error) for (const j of jobs) this.settle(j, () => j.reject(new Error(ev.data.error)));
        else if (ev.data.tiles) jobs.forEach((j, k) => this.settle(j, () => j.resolve({ ...ev.data, tile: ev.data.tiles[k] })));
        else this.settle(jobs[0], () => jobs[0].resolve(ev.data));
        this.pump();
      };
      this.workers.push({ w, busy: 0, shards: new Set(), regions: new Set(), coverings: new Set() });
    }
  }

  get size() {
    return this.workers.length;
  }

  /** Requests queued or running. */
  get inFlight() {
    let n = 0;
    for (const o of this.owners.values()) n += o.count;
    return n;
  }

  inFlightOf(owner: number): number {
    return this.owners.get(owner)?.count ?? 0;
  }

  /** A source's jobs are counted and its fetched bytes reported here. */
  register(owner: number, fetched: (bytes: number, requests: number) => void) {
    this.owners.set(owner, { count: 0, fetched });
  }

  unregister(owner: number) {
    this.owners.delete(owner);
  }

  private settle(job: Job, fn: () => void) {
    if (job.state === "done") return;
    job.state = "done";
    const o = this.owners.get(job.owner);
    if (o) o.count--;
    fn();
  }

  submit(j: Pick<Job, "owner" | "build" | "shard" | "url" | "ranges" | "dataEnd" | "region"> & Partial<Pick<Job, "urgent" | "affinity" | "tile">>): { job: Job; promise: Promise<any> } {
    let job!: Job;
    const promise = new Promise<any>((resolve, reject) => {
      job = { ...j, resolve, reject, waiters: 1, state: "queued", urgent: j.urgent ?? false, batch: 0, seq: 0 };
    });
    const o = this.owners.get(job.owner);
    if (o) o.count++;
    this.stamp(job);
    (job.urgent ? this.urgent : this.queue).push(job);
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
    if (job.state !== "queued") {
      job.urgent ||= urgent;
      return;
    }
    const from = job.urgent ? this.urgent : this.queue;
    const i = from.indexOf(job);
    if (i < 0) return;
    from.splice(i, 1);
    job.urgent ||= urgent;
    this.stamp(job);
    (job.urgent ? this.urgent : this.queue).push(job);
    this.schedule();
  }

  /** Drop a job no one waits for: one not yet started, or one whose
   * fetch no other waited-for job shares (the fetch is aborted). */
  cancel(job: Job): boolean {
    const abort = new DOMException("tile no longer needed", "AbortError");
    if (job.state === "queued") {
      const line = job.urgent ? this.urgent : this.queue;
      const i = line.indexOf(job);
      if (i >= 0) line.splice(i, 1);
      this.settle(job, () => job.reject(abort));
      return true;
    }
    const g = job.group;
    if (!g || g.cancelled || g.jobs.some((j) => j.waiters > 0)) return false;
    if (job.state === "fetching") {
      g.cancelled = true;
      for (const a of g.aborts) a.abort();
      return true;
    }
    if (job.state === "ready") {
      const i = this.ready.indexOf(g);
      if (i >= 0) this.ready.splice(i, 1);
      g.cancelled = true;
      for (const j of g.jobs) this.settle(j, () => j.reject(abort));
      return true;
    }
    return false;
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

  /** Whether `a` goes before `b`: urgent, the newest batch, the first asked in it. */
  private static before(a: Job, b: Job): boolean {
    if (a.urgent !== b.urgent) return a.urgent;
    return a.batch > b.batch || (a.batch === b.batch && a.seq < b.seq);
  }

  /** Take the next job of `line`: the newest batch, the first asked in it. */
  private take(line: Job[]): Job {
    let best = 0;
    for (let i = 1; i < line.length; i++) if (Pool.before(line[i], line[best])) best = i;
    return line.splice(best, 1)[0];
  }

  private takeGroup(): Group {
    let best = 0;
    for (let i = 1; i < this.ready.length; i++) if (Pool.before(this.ready[i].jobs[0], this.ready[best].jobs[0])) best = i;
    return this.ready.splice(best, 1)[0];
  }

  private affine = new Map<string, number>();

  private pump() {
    for (;;) {
      if (!this.decode() && !this.startFetch()) return;
    }
  }

  /** A fetched group to a worker with a free slot. */
  private decode(): boolean {
    if (!this.ready.length) return false;
    let slot = -1;
    for (let i = 0; i < this.workers.length; i++) {
      if (this.workers[i].busy < Pool.PER_WORKER && (slot < 0 || this.workers[i].busy < this.workers[slot].busy)) slot = i;
    }
    if (slot < 0) return false;
    const g = this.takeGroup();
    const job = g.jobs[0];
    if (job.affinity != null) {
      const pref = this.affine.get(job.affinity);
      if (pref != null && this.workers[pref].busy < Pool.PER_WORKER) slot = pref;
      if (this.affine.size > 4096) this.affine.clear();
      this.affine.set(job.affinity, slot);
    }
    for (const j of g.jobs) j.state = "running";
    const w = this.workers[slot];
    const id = this.next++;
    const needsFooter = !w.shards.has(job.shard);
    w.shards.add(job.shard);
    w.busy++;
    this.running.set(id, { group: g, slot });
    // a kept block is copied, the group's own pieces transferred
    const pieces = g.pieces.map((p, k) => (g.owned[k] ? p : { ...p, buf: p.buf.slice(0) }));
    const msg = g.jobs.length > 1 ? job.tile!.merge(id, needsFooter, g.jobs.map((j) => j.tile!.n), pieces) : job.build(id, needsFooter, pieces);
    // the region, by id; its rings with a worker's first request naming it,
    // and the coverings other workers computed that this one lacks
    if (job.region) {
      msg.region = job.region.id;
      if (!w.regions.has(job.region.id)) {
        msg.regionRings = job.region.rings;
        w.regions.add(job.region.id);
      }
      const known = coverings.get(job.region.id);
      if (known) {
        const give: Covering[] = [];
        for (const [key, ranges] of known) {
          const k = `${job.region.id}|${key}`;
          if (w.coverings.has(k)) continue;
          w.coverings.add(k);
          give.push({ region: job.region.id, key, ranges });
        }
        if (give.length) msg.regionCoverings = give;
      }
    }
    w.w.postMessage(msg, pieces.map((p) => p.buf));
    return true;
  }

  /** The next queued job (and the queued tiles of its row group) to the network. */
  private startFetch(): boolean {
    const line = this.urgent.length ? this.urgent : this.queue;
    if (!line.length) return false;
    const urgent = line === this.urgent;
    if (urgent ? this.fetchingUrgent >= this.connections + URGENT_CONNECTIONS : this.fetching >= this.connections) return false;
    const job = this.take(line);
    const jobs = [job];
    if (job.tile && job.affinity != null) {
      for (let i = 0; i < line.length && jobs.length < Pool.MERGE; ) {
        if (line[i].affinity === job.affinity && line[i].tile) jobs.push(line.splice(i, 1)[0]);
        else i++;
      }
    }
    const g: Group = { jobs, pieces: [], owned: [], aborts: [], cancelled: false };
    for (const j of jobs) {
      j.state = "fetching";
      j.group = g;
    }
    if (!job.ranges) {
      g.jobs.forEach((j) => (j.state = "ready"));
      this.ready.push(g);
      return true;
    }
    if (urgent) this.fetchingUrgent++;
    else this.fetching++;
    const o = this.owners.get(job.owner);
    let bytes = 0, requests = 0;
    const count = (b: number) => ((bytes += b), requests++);
    const wanted = jobs.flatMap((j) => j.ranges!);
    const { ranges, blocks } = this.plan(wanted, job.dataEnd);
    g.aborts = ranges.map(() => new AbortController());
    Promise.all([
      ...ranges.map((r, k) => this.fetchRange(job.url, r, g.aborts[k].signal, count).then((buf) => ({ lo: r[0], hi: r[1], buf, owned: true }))),
      ...blocks.map((r) => this.fetchBlock(job.url, r, count).then((buf) => ({ lo: r[0], hi: r[1], buf, owned: false }))),
    ]).then(
      (pieces) => {
        if (urgent) this.fetchingUrgent--;
        else this.fetching--;
        o?.fetched(bytes, requests);
        if (g.cancelled) {
          for (const j of jobs) this.settle(j, () => j.reject(new DOMException("tile no longer needed", "AbortError")));
        } else {
          g.pieces = pieces;
          g.owned = pieces.map((p) => p.owned);
          for (const j of jobs) j.state = "ready";
          this.ready.push(g);
        }
        this.pump();
      },
      (e) => {
        if (urgent) this.fetchingUrgent--;
        else this.fetching--;
        o?.fetched(bytes, requests);
        const err = g.cancelled ? new DOMException("tile no longer needed", "AbortError") : (e as Error);
        for (const j of jobs) this.settle(j, () => j.reject(err));
        this.pump();
      },
    );
    return true;
  }

  /**
   * What to fetch for `wanted` ranges: small ones as the blocks holding
   * them (kept across jobs), the rest coalesced where they touch or
   * overlap, all before `dataEnd`.
   */
  private plan(wanted: [number, number][], dataEnd: number): { ranges: [number, number][]; blocks: [number, number][] } {
    const B = this.block;
    const large: [number, number][] = [];
    const blocks = new Map<number, [number, number]>();
    for (const [lo, hi] of wanted) {
      if (hi <= lo) continue;
      if (B > 0 && hi - lo <= B) {
        const a = Math.floor(lo / B) * B;
        const b = Math.min(Math.ceil(hi / B) * B, Math.max(dataEnd, hi));
        blocks.set(a, [a, b]);
      } else large.push([lo, hi]);
    }
    large.sort((x, y) => x[0] - y[0]);
    const ranges: [number, number][] = [];
    for (const r of large) {
      const last = ranges[ranges.length - 1];
      if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
      else ranges.push([r[0], r[1]]);
    }
    return { ranges, blocks: [...blocks.values()] };
  }

  private async fetchRange(url: string, [lo, hi]: [number, number], signal: AbortSignal | undefined, count: (b: number) => void): Promise<ArrayBuffer> {
    const res = await fetch(url, { headers: { Range: `bytes=${lo}-${hi - 1}` }, signal });
    if (res.status !== 206 && res.status !== 200) throw new Error(`${url}: HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    count(buf.byteLength);
    // a server that ignores Range sends the whole file
    return res.status === 200 && buf.byteLength > hi - lo ? buf.slice(lo, hi) : buf;
  }

  /** A block, fetched once and kept (the most recently used last). */
  private fetchBlock(url: string, r: [number, number], count: (b: number) => void): Promise<ArrayBuffer> {
    const key = `${url}#${r[0]}-${r[1]}`;
    const hit = this.blocks.get(key);
    if (hit) {
      this.blocks.delete(key);
      this.blocks.set(key, hit);
      return hit;
    }
    const p = this.fetchRange(url, r, undefined, count);
    this.blocks.set(key, p);
    p.catch(() => this.blocks.delete(key));
    while (this.blocks.size > Pool.BLOCKS) this.blocks.delete(this.blocks.keys().next().value!);
    return p;
  }

  /** Forget a file's kept blocks (its URL has changed). */
  forget(url: string) {
    for (const k of [...this.blocks.keys()]) if (k.startsWith(url + "#")) this.blocks.delete(k);
  }

  terminate() {
    for (const { w } of this.workers) w.terminate();
    const closed = () => new Error("pyramid closed");
    for (const { group } of this.running.values()) for (const job of group.jobs) this.settle(job, () => job.reject(closed()));
    for (const g of this.ready) for (const job of g.jobs) this.settle(job, () => job.reject(closed()));
    for (const job of [...this.queue, ...this.urgent]) this.settle(job, () => job.reject(closed()));
    this.running.clear();
    this.ready = [];
    this.queue = [];
    this.urgent = [];
    this.blocks.clear();
  }
}

/**
 * A pool of decode workers for several sources to share (`pool` in
 * SourceOptions), terminated by the caller. `workers` as in SourceOptions;
 * `connections` and `block` as in ReadTuning.
 */
export function createWorkerPool(opts: { workers?: number; worker?: WorkerFactory; connections?: number; block?: number } = {}): WorkerPool {
  const workers = opts.workers ?? Math.max(2, Math.min(4, (globalThis.navigator?.hardwareConcurrency || 4) - 2));
  return new Pool(workers, opts.worker ?? defaultWorker, opts.connections, opts.block);
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
    statsMs: 0, cached: 0, cachedBytes: 0, inFlight: 0, tilesSkipped: 0, rowsDropped: 0, footerMs: 0, indexMs: 0, tileRequests: 0,
  };
  /** The set's metadata from its writer (the `a5pyramids` entry), parsed. */
  readonly meta: Record<string, any>;
  /** Whether tiles come with PCA sums: the `pca` option, else the set's colour is "pca". */
  readonly pca: boolean;
  /** The region's rings (the `region` option), or null. */
  readonly region: Ring[] | null;
  /** Whether tiles of one row group share a request (the `merge` option or its default). */
  readonly merge: boolean;
  private readonly regionBoxes: Box[] | null;
  private readonly regionRef: { id: number; rings: Ring[] } | null;

  private shards: ShardState[];
  private byPath = new Map<string, ShardState>();
  private cache = new Map<string, Promise<PackedTile>>();
  private jobs = new Map<string, Job>();
  private sizes = new Map<string, number>();
  private pool: Pool;
  private readonly ownsPool: boolean;

  private readonly refreshSet?: () => Promise<SetIndex>;
  private refreshing: Promise<boolean> | null = null;
  private readonly read: ReadTuning["read"];

  private constructor(readonly base: string, public set: SetIndex, workers: number, readonly cacheBytes: number, worker: WorkerFactory,
    opts: SourceOptions = {}) {
    this.read = opts.read ?? {};
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
    this.pca = opts.pca ?? this.meta.colour === "pca";
    // shards listed as empty have no file
    this.shards = set.shards
      .filter((e) => e.path != null || e.url != null)
      .map((entry) => ({ entry, key: shardKey(entry), url: entry.url ?? new URL(entry.path!, base).href }));
    for (const s of this.shards) this.byPath.set(s.key, s);
    this.merge = opts.merge ?? !this.shards.some((s) => /^https?:/i.test(s.url));
    this.region = opts.region ? checkRings(opts.region) : null;
    this.regionBoxes = this.region ? this.region.map(ringBox) : null;
    this.regionRef = this.region ? { id: regionId(this.region), rings: this.region } : null;
    if (opts.pool) {
      if (!(opts.pool instanceof Pool)) throw new Error("pool: give one from createWorkerPool()");
      this.pool = opts.pool;
      this.ownsPool = false;
    } else {
      this.pool = new Pool(workers, worker, opts.connections, opts.block);
      this.ownsPool = true;
    }
    this.pool.register(this.id, (bytes, requests) => {
      this.stats.tileBytes += bytes;
      this.stats.tileRequests += requests;
    });
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
          this.pool.forget(s.url);
          s.url = url;
          if (!same) {
            s.entry = e;
            s.index = s.footer = s.tileRows = s.tileBytes = s.tileGroup = s.tileRanges = s.rangeStarts = s.tileMeets = s.opening = undefined;
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

  /** This source's requests sent to the pool and not yet answered. */
  get inFlight() {
    return this.pool.inFlightOf(this.id) + this.footersOpening;
  }
  /** Footers being read (before their index reaches a worker). */
  private footersOpening = 0;

  /** Drop the cache and, unless the pool is shared, its workers. */
  close() {
    this.pool.unregister(this.id);
    if (this.ownsPool) this.pool.terminate();
    this.cache.clear();
  }

  tierOf(lod: number) {
    return tierOf(this.set, lod);
  }

  /** Shards of `tier` meeting `box` (and the region's rings' boxes), open or not. */
  shardsIn(tier: number, box: Box): ShardState[] {
    return this.shards.filter((s) => s.entry.tier === tier && s.entry.bbox != null && boxesMeet(s.entry.bbox, box) && this.inRegion(s.entry.bbox));
  }

  /** Whether a box meets the region's (every box, without one). */
  private inRegion(box: Box): boolean {
    return !this.regionBoxes || this.regionBoxes.some((b) => boxesMeet(b, box));
  }

  /** Start opening the shards of `tier` meeting `box`; resolves when all have. */
  ensure(tier: number, box: Box): Promise<void> {
    this.cover(tier);
    return Promise.all(this.shardsIn(tier, box).map((s) => this.openShard(s))).then(() => undefined);
  }

  /**
   * With a region: its coverings of the tier's levels, computed in a worker
   * while the tier's footers are in flight (an index needs them to tell
   * which tiles meet the region; a tile to drop its rows outside), once a
   * tier, unless every worker has them already.
   */
  private covered = new Set<number>();
  private cover(tier: number) {
    const r = this.regionRef;
    if (!r || this.covered.has(tier)) return;
    this.covered.add(tier);
    const t = this.set.tiers[tier];
    if (!t) return;
    const known = coverings.get(r.id);
    const lods: number[] = [];
    for (let lod = t.lods[0]; lod <= t.lods[1]; lod++) {
      if (known && this.region!.every((_, i) => known.has(`${lod}|${i}`))) continue;
      lods.push(lod);
    }
    if (!lods.length) return;
    // the finest first: the level a view wants, then its backdrop
    lods.reverse();
    this.pool.submit({
      owner: this.id, build: (id) => ({ id, type: "cover", lods }),
      shard: `cover#${r.id}`, url: "", ranges: null, dataEnd: 0, region: r, urgent: true,
    }).promise.then((out) => (this.stats.indexMs += out.ms ?? 0), () => {});
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
        const t0 = performance.now();
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
        this.stats.footerMs += performance.now() - t0;
        s.footer = footer;
        const url = s.url;
        const out = await this.pool.submit({
          owner: this.id,
          build: (id) => ({ id, type: "index", url, bytes, footer: footer.slice(0), columns: this.columns, vector: this.vector }),
          shard: url, url, ranges: null, dataEnd: bytes - footer_bytes, region: this.regionRef,
          urgent: true, // a shard's index decides which tiles exist: before any tile
        }).promise;
        this.stats.indexMs += out.ms ?? 0;
        const index = checkIndex(JSON.parse(out.kv.a5pyramid), out.kv.a5_index, s.key);
        const n = index.tile_bbox.length / 4;
        if (out.tileRows.length !== n) throw new Error(`the index has ${n} tiles, the footer ${out.tileRows.length}`);
        s.tileRows = out.tileRows;
        s.tileBytes = out.tileBytes;
        s.tileGroup = out.tileGroup;
        s.tileRanges = out.ranges;
        s.rangeStarts = out.starts;
        if (out.tileMeets) {
          s.tileMeets = out.tileMeets as Uint8Array;
          for (const m of s.tileMeets) if (!m) this.stats.tilesSkipped++;
        }
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
        if (s.tileMeets && !s.tileMeets[i]) continue;
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
      for (const s of this.shards) if (s.entry.tier === t && (s.entry.bbox == null || this.inRegion(s.entry.bbox))) (rows += s.entry.rows), (bytes += s.entry.bytes);
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
      this.stats.readMs += t.readMs;
      this.stats.decodeMs += t.decodeMs ?? 0;
      this.stats.packMs += t.packMs;
      this.stats.statsMs += t.statsMs;
      this.stats.rowsDropped += t.dropped ?? 0;
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

  /** Tile `i`'s byte ranges, from the shard's index. */
  private rangesOf(s: ShardState, i: number): [number, number][] {
    const r = s.tileRanges, st = s.rangeStarts;
    if (!r || !st || i + 1 >= st.length) return [[0, s.entry.bytes - s.entry.footer_bytes]];
    const out: [number, number][] = [];
    for (let k = st[i]; k < st[i + 1]; k++) out.push([r[2 * k], r[2 * k + 1]]);
    return out;
  }

  /** Queue the read of tile `i` of `s`; tiles of one row group can share a request. */
  private submitTile(s: ShardState, i: number, lod: number, urgent: boolean) {
    const footer = s.footer!;
    const read: ReadOptions = { ...this.read };
    const group = s.tileGroup?.[i] ?? -1;
    const url = s.url;
    const request = (id: number, needsFooter: boolean, tiles: number[], pieces: Piece[]): WorkerRequest => ({
      id, type: "decode", url, bytes: s.entry.bytes, footer: needsFooter ? footer.slice(0) : undefined,
      tile: tiles[0], tiles: tiles.length > 1 ? tiles : undefined, lod, columns: this.columns, stats: this.pca, vector: this.vector, read, pieces,
    });
    return this.pool.submit({
      owner: this.id,
      build: (id, needsFooter, pieces) => request(id, needsFooter, [i], pieces),
      shard: url, url, ranges: this.rangesOf(s, i), dataEnd: s.entry.bytes - s.entry.footer_bytes, region: this.regionRef,
      urgent,
      affinity: group >= 0 ? `${url}#${group}` : undefined,
      tile: group >= 0 && this.merge ? { n: i, merge: request } : undefined,
    });
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
