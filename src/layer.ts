// A5PyramidLayer: a deck.gl layer drawing a pyramid set, ported from a5view's
// lazy renderer (inst/htmlwidgets/lib/a5view-tiling/lazy-renderer.js).
//
// As in a5view:
//   - each LOD draws through its own TileLayer, whose Tileset2D has the
//     level's tiles as atomic tiles with the boxes from the index, and no
//     parent walk (A5 is not the quadtree deck.gl assumes);
//   - each tile is one SolidPolygonLayer over binary attributes (boundaries
//     from a5-js, packed in a worker), never PolygonLayer, which draws nothing
//     from binary data;
//   - on a LOD change the new level builds undrawn and replaces the old one
//     in the first frame that every tile it selected is loaded, or after
//     MAX_HOLD_MS once at least half of them are (a5view: after MAX_HOLD_MS
//     whatever is loaded, which blanks the map on a slow link), so the map
//     neither blanks nor draws two levels at once.
// Different from a5view:
//   - the tiles of a LOD come from every open shard of its tier; shards open
//     when a view first meets them, and a new version of the source redoes
//     tile selection;
//   - getTileData returns the decode's promise, so deck.gl tracks loading
//     (a5view decoded outside deck and marked tiles dirty);
//   - cells finer than FAN_MIN_LOD come with fan triangles, so deck.gl skips
//     earcut, its largest per-tile cost in a5view;
//   - colours are computed on the GPU from the codes each tile uploads once
//     (codes-layer.ts), through a Colouring of a few uniforms (a band
//     stretch, or the PCA of the view), so new colours redraw without
//     reading, packing or uploading a tile again;
//   - the LOD is the coarsest whose cells are at most `cellPx` CSS pixels
//     across at the view's centre (tiling.ts), not a5view's zoom table,
//     made coarser while the tiles it would read for the view exceed a
//     budget of bytes or cells (their sizes come with each shard's footer),
//     so a view sparse in data is drawn finer than a full one;
//   - a backdrop level, BACKDROP_STEP LODs coarser than the one wanted (a
//     sixteenth of its cells at 2), is read ahead of every other tile and
//     drawn under a level still loading, so a view is covered, coarsely,
//     after one or two small reads;
//   - a base level, the finest whose tiles over the whole set fit
//     BASE_BUDGET (LOD 12 for Sabah: 38 k cells, under 3 MB of the 64-band
//     set), is read once when the set opens, kept, and drawn under the rest
//     while they load, so a jump to any part of the set shows it at once;
//   - the base is drawn only where the levels above have no loaded tile (a
//     Cover of their id runs, codes-layer.ts), so its large cells fill gaps
//     without standing out past a coast.

import { CompositeLayer, type GetPickingInfoParams, type Layer, type LayerProps, type PickingInfo, type UpdateParameters } from "@deck.gl/core";
import { TileLayer, _Tileset2D as Tileset2D } from "@deck.gl/geo-layers";
import { CodesPolygonLayer, cellKey, type Cover, type Palette } from "./codes-layer.js";
import { BASE_BUDGET, DEFAULT_BUDGET, chooseLevel, type Budget } from "./budget.js";
import { boxesMeet, type Box } from "./format.js";
import { a5Resolution, boxKey, pickLod, viewportBox } from "./tiling.js";
import type { Colouring } from "./colour.js";
import { WORLD, type PyramidSource } from "./source.js";
import type { PackedTile } from "./tile-core.js";

export const MAX_HOLD_MS = 1500;
/** LODs between the level wanted and its backdrop. */
export const BACKDROP_STEP = 2;

export interface CellInfo {
  cell: bigint;
  lod: number;
  ncell: number;
  values: number[];
}

export interface RenderStats {
  lod: number | null;
  /** The level shown has every tile in view, every shard of its tier there opened. */
  ready: boolean;
  /** The level the pixel target asks for, and why a coarser one is drawn. */
  targetLod: number | null;
  capped: "bytes" | "cells" | null;
  /** What the level chosen reads for the view (null until its shards open). */
  viewBytes: number | null;
  viewCells: number | null;
  tier: number;
  pendingLod: number | null;
  tilesSelected: number;
  tilesLoaded: number;
  cellsDrawn: number;
  /** The loaded tiles of the LOD shown, in view. */
  tiles: PackedTile[];
  /** The backdrop level, its loaded tiles in view, and whether it is complete. */
  backdropLod: number | null;
  backdrop: PackedTile[];
  backdropReady: boolean;
}

export interface A5PyramidLayerProps extends LayerProps {
  source: PyramidSource;
  colouring: Colouring;
  /** Cells drawn at most this many CSS pixels across (the square root of their area). */
  cellPx?: number;
  /** The most a view reads and draws; coarser levels past it. */
  budget?: Budget;
  /**
   * Called with what the layer draws (a fresh object each time) after it
   * changes: the levels, their tiles, the view's cost. Called outside
   * deck.gl's render, at most once a task.
   */
  onStats?: (stats: RenderStats) => void;
}

/** Whether a view has settled: no level pending, the level shown ready
 * (every tile in view loaded, its shards opened), nothing in flight. */
export function isSettled(stats: RenderStats, source: PyramidSource): boolean {
  return stats.ready && stats.pendingLod == null && source.inFlight === 0;
}

export function emptyRenderStats(): RenderStats {
  return {
    lod: null, ready: false, targetLod: null, capped: null, viewBytes: null, viewCells: null, tier: 0, pendingLod: null, tilesSelected: 0, tilesLoaded: 0, cellsDrawn: 0, tiles: [],
    backdropLod: null, backdrop: [], backdropReady: false,
  };
}

const tileIds = new WeakMap<PackedTile, number>();
let nextTileId = 1;
function tileId(t: PackedTile): number {
  let id = tileIds.get(t);
  if (id == null) tileIds.set(t, (id = nextTileId++));
  return id;
}

/** Shared by the parent and its TileLayers: which LOD is drawn. */
class Swap {
  shown: number | null = null;
  pending: number | null = null;
  backdrop: number | null = null;
  base: number | null = null;
  pendingSince = 0;
  ready = new Map<number, boolean>();
  loaded = new Map<number, { selected: number; loaded: number; cells: number }>();
  contents = new Map<number, PackedTile[]>();
  redraw: () => void = () => {};
  private queued = false;

  /**
   * Whether the pending level `lod` may replace the shown one: complete, or
   * held for MAX_HOLD_MS with at least half its tiles loaded, never empty.
   */
  canShow(lod: number, now: number): boolean {
    if (this.ready.get(lod)) return true;
    if (now - this.pendingSince < MAX_HOLD_MS) return false;
    const c = this.loaded.get(lod);
    return !!c && c.selected > 0 && 2 * c.loaded >= c.selected;
  }

  draws(lod: number): boolean {
    const top = this.pending != null && this.ready.get(this.pending) ? this.pending : this.shown;
    if (lod === top) return true;
    // the backdrop, under a level still loading
    return lod === this.backdrop && top != null && !this.ready.get(top);
  }

  /** What hides the base's cells: the loaded runs of the levels drawn
   * above it; changed in place. */
  baseCover: Cover = { version: 0, runs: new Map() };
  private coverKey = "";

  updateCovers() {
    const top = this.pending != null && this.ready.get(this.pending) ? this.pending : null;
    const lods = [...new Set([this.shown, top, this.backdrop])].filter((l): l is number => l != null && (this.base == null || l > this.base));
    const key = lods.map((l) => `${l}:` + (this.contents.get(l) ?? []).map(tileId).join(",")).join("|");
    if (key === this.coverKey) return;
    this.coverKey = key;
    const runs = new Map<number, Float64Array>();
    for (const l of lods) {
      const ts = this.contents.get(l) ?? [];
      if (!ts.length) continue;
      const pairs = ts.map((t) => [cellKey(t.cells[0]), cellKey(t.cells[t.n - 1])]).sort((a, b) => a[0] - b[0]);
      runs.set(l, Float64Array.from(pairs.flat()));
    }
    this.baseCover.runs = runs;
    this.baseCover.version++;
  }

  /** The base level is drawn under a finer level still loading. */
  drawsBase(): boolean {
    const top = this.pending != null && this.ready.get(this.pending) ? this.pending : this.shown;
    return this.base != null && top != null && top > this.base && !this.ready.get(top);
  }

  sink: RenderStats | null = null;
  /** Tell the parent the stats may have changed (it batches). */
  notify: () => void = () => {};

  report(lod: number, ready: boolean, counts: { selected: number; loaded: number; cells: number }, contents: PackedTile[]) {
    const was = this.ready.get(lod);
    const before = this.loaded.get(lod)?.loaded;
    this.ready.set(lod, ready);
    this.loaded.set(lod, counts);
    this.contents.set(lod, contents);
    if (this.sink && lod === this.shown) {
      this.sink.ready = ready;
      this.sink.tilesSelected = counts.selected;
      this.sink.tilesLoaded = counts.loaded;
      this.sink.cellsDrawn = counts.cells;
      this.sink.tiles = contents;
    }
    if (this.sink && lod === this.backdrop) {
      this.sink.backdrop = contents;
      this.sink.backdropReady = ready;
    }
    this.updateCovers();
    this.notify();
    // the pending level completed, or gained a tile after the hold: let the
    // parent decide whether to drop the old one, through a timer, not from
    // inside deck's render pass
    if (lod === this.pending && ((ready && !was) || (counts.loaded !== before && performance.now() - this.pendingSince >= MAX_HOLD_MS))) this.later();
  }

  later() {
    if (this.queued) return;
    this.queued = true;
    setTimeout(() => {
      this.queued = false;
      this.redraw();
    }, 0);
  }

  /** Redraw after `ms`, so a level that never completes is shown anyway. */
  hold(ms: number) {
    setTimeout(() => this.redraw(), ms);
  }
}

interface PyramidTileLayerProps {
  /** "base": the base level's layer, which neither reports nor holds. */
  role?: "base";
  lod: number;
  tier: number;
  swap: Swap;
  source: PyramidSource;
  version: number;
}

/** A TileLayer that draws only when its LOD is the one shown, and reports
 * whether every tile it selected has loaded. */
class PyramidTileLayer extends TileLayer<PackedTile, PyramidTileLayerProps> {
  static layerName = "A5PyramidTileLayer";

  renderLayers() {
    const out = super.renderLayers();
    const tiles = this.state.tileset?.selectedTiles ?? [];
    let loaded = 0;
    let cells = 0;
    const contents: PackedTile[] = [];
    for (const t of tiles) {
      if (t.isLoaded) {
        loaded++;
        const c = t.content as PackedTile | null;
        cells += c?.n ?? 0;
        if (c && c.n > 0) contents.push(c);
      }
    }
    // complete once every tile selected has loaded and every shard of the
    // tier in view has opened (before that a level selects too few tiles)
    const { source, tier, lod, swap, role } = this.props;
    if (role === "base") return out;
    const opened = source.ready(lod, viewportBox(this.context.viewport));
    // (the tiles' own flags: the layer's isLoaded also waits for sublayers
    // made in this very pass, so a level's last tile would never count)
    swap.report(lod, opened && !!this.state.tileset && loaded === tiles.length, { selected: tiles.length, loaded, cells }, contents);
    return out;
  }

  filterSubLayer(context: Parameters<TileLayer["filterSubLayer"]>[0]) {
    const { swap, lod, role } = this.props;
    return (role === "base" ? swap.drawsBase() : swap.draws(lod)) && super.filterSubLayer(context);
  }
}

function tilesetClass(source: PyramidSource, lod: number, swap: Swap, all = false): typeof Tileset2D {
  return class extends Tileset2D {
    private lastKey: string | null = null;
    private lastIndices: { i: string }[] = [];

    private heldKey: string | null = null;
    private heldIndices: { i: string }[] = [];

    getTileIndices(opts: { viewport: any }) {
      // the base level: every tile, wherever the view is
      if (all) {
        const key = `all|${source.version}`;
        if (key !== this.lastKey) {
          this.lastKey = key;
          this.lastIndices = source.tilesIn(lod, WORLD).map((t) => ({ i: t.key }));
        }
        return this.lastIndices as any;
      }
      const box = viewportBox(opts.viewport);
      const key = boxKey(box) + "|" + source.version;
      // a level held on screen while another loads adds no tiles: it would
      // otherwise read tiles for a view it will not be shown in (after a
      // zoom out, a hundred of them). It keeps those of its tiles still in
      // view, and drops (and so aborts) the rest: a flight can leave it
      // holding a selection made far from where the view ends.
      if (this.lastKey != null && swap.pending != null && lod !== swap.pending && lod !== swap.backdrop) {
        if (key !== this.heldKey) {
          this.heldKey = key;
          this.heldIndices = this.lastIndices.filter((t) => {
            const b = source.tileBox(t.i);
            return !b || boxesMeet(b, box);
          });
        }
        return this.heldIndices as any;
      }
      this.heldKey = null;
      if (key !== this.lastKey) {
        this.lastKey = key;
        // nothing until every shard of the tier in view has opened: the
        // budget cannot judge the level before, and a level it then turns
        // down would already be reading
        this.lastIndices = source.ready(lod, box) ? source.tilesIn(lod, box).map((t) => ({ i: t.key })) : [];
      }
      return this.lastIndices as any;
    }
    getTileId(index: any) {
      return index ? index.i : null;
    }
    // deck.gl aborts a loading tile that left the view only once more than
    // maxRequests are loading, so after a pan its tiles stay queued; abort
    // them at once, as deck.gl-raster's abortUnwantedLoads does (the pool
    // drops a job not yet started)
    update(...args: Parameters<Tileset2D["update"]>) {
      const out = super.update(...args);
      for (const tile of (this as any)._cache.values()) {
        if (tile.isLoading && !tile.isSelected && !tile.isVisible) tile.abort();
      }
      return out;
    }
    getTileMetadata(index: any) {
      let b = source.tileBox(index.i) ?? [-180, -85.05, 180, 85.05];
      // deck.gl wants west <= east; getTileIndices culls exactly
      if (b[0] > b[2]) b = [-180, b[1], 180, b[3]];
      return { bbox: { west: b[0], south: b[1], east: b[2], north: b[3] } };
    }
    getTileZoom() {
      return lod;
    }
    // no parents: A5 tiles are atomic (a5view overrides both, and deck.gl's
    // tree rebuild walks ancestors through _getNearestAncestor)
    getParentIndex() {
      return null as any;
    }
    _getNearestAncestor() {
      return null as any;
    }
  };
}

export class A5PyramidLayer extends CompositeLayer<A5PyramidLayerProps> {
  static layerName = "A5PyramidLayer";
  static defaultProps = { cellPx: 1, budget: DEFAULT_BUDGET, pickable: true };

  declare state: {
    swap: Swap;
    tilesets: Map<number, typeof Tileset2D>;
    datas: WeakMap<PackedTile, unknown>;
    palette: Palette;
    baseTileset: typeof Tileset2D | null;
    stats: RenderStats;
    statsKey: string;
    statsQueued: boolean;
  };

  initializeState() {
    const swap = new Swap();
    swap.redraw = () => this.setNeedsUpdate();
    this.setState({
      swap, tilesets: new Map(), datas: new WeakMap(), palette: { colouring: this.props.colouring }, baseTileset: null,
      stats: emptyRenderStats(), statsKey: "", statsQueued: false,
    });
    swap.sink = this.state.stats;
    swap.notify = () => this.queueStats();
    this.props.source.onChange = () => swap.later();
    this.loadBase(this.props.source);
  }

  /** Send the stats to onStats after this task, if they changed. */
  private queueStats() {
    if (this.state.statsQueued || !this.props.onStats) return;
    this.state.statsQueued = true;
    setTimeout(() => {
      if (!this.state) return;
      this.state.statsQueued = false;
      const r = this.state.stats;
      const key = [
        r.lod, r.ready, r.pendingLod, r.targetLod, r.capped, r.viewBytes, r.viewCells, r.tilesSelected, r.tilesLoaded, r.cellsDrawn,
        r.backdropLod, r.backdrop.length, r.backdropReady, r.tiles.length,
      ].join("|");
      if (key === this.state.statsKey) return;
      this.state.statsKey = key;
      this.props.onStats?.({ ...r });
    }, 0);
  }

  /** Find the source's base level; its layer joins once it is known. */
  private loadBase(source: PyramidSource) {
    const swap = this.state.swap;
    swap.base = null;
    source.baseLevel(BASE_BUDGET).then(
      (lod) => {
        if (this.props.source !== source) return;
        swap.base = lod;
        swap.later();
      },
      () => {},
    );
  }

  shouldUpdateState({ changeFlags }: UpdateParameters<this>) {
    return changeFlags.somethingChanged;
  }

  updateState({ props, oldProps }: UpdateParameters<this>) {
    // every tile's layer reads the palette as it draws: new colours are a redraw
    if (props.colouring !== this.state.palette.colouring) {
      this.state.palette.colouring = props.colouring;
      this.setNeedsRedraw();
    }
    if (props.source !== oldProps.source) {
      this.state.tilesets.clear();
      this.state.swap.shown = null;
      this.state.swap.pending = null;
      this.state.swap.backdrop = null;
      this.state.swap.contents.clear();
      this.state.swap.ready.clear();
      this.state.swap.loaded.clear();
      this.state.baseTileset = null;
      this.state.stats = emptyRenderStats();
      this.state.statsKey = "";
      this.state.swap.sink = this.state.stats;
      props.source.onChange = () => this.state.swap.later();
      this.loadBase(props.source);
    }
  }

  getPickingInfo({ info }: GetPickingInfoParams): PickingInfo {
    const t = (info as any).sourceTile?.content as PackedTile | undefined;
    if (t && info.index >= 0 && info.index < t.n) {
      (info as any).object = {
        cell: t.cells[info.index],
        lod: t.lod,
        ncell: t.ncell[info.index],
        values: t.vector
          ? Array.from(t.vector.subarray(info.index * t.dim, (info.index + 1) * t.dim))
          : t.values.map((v) => v[info.index]),
      } satisfies CellInfo;
    }
    return info;
  }

  private tileset(lod: number) {
    let cls = this.state.tilesets.get(lod);
    if (!cls) {
      cls = tilesetClass(this.props.source, lod, this.state.swap);
      this.state.tilesets.set(lod, cls);
    }
    return cls;
  }

  private tileLayer(lod: number, urgent = false, base = false) {
    const { source, opacity, pickable } = this.props;
    const { datas, palette } = this.state;
    let TilesetClass = this.tileset(lod);
    if (base) {
      this.state.baseTileset ??= tilesetClass(source, lod, this.state.swap, true);
      TilesetClass = this.state.baseTileset;
    }
    return new PyramidTileLayer(
      this.getSubLayerProps({
        // per source: deck keeps a TileLayer's tileset across prop changes,
        // so a new source needs new layers, not new props
        id: base ? `s${source.id}-base${lod}` : `s${source.id}-lod${lod}`,
        role: base ? "base" : undefined,
        lod,
        tier: source.tierOf(lod),
        source,
        swap: this.state.swap,
        version: source.version,
        TilesetClass,
        extent: [-180, -85.05, 180, 85.05],
        refinementStrategy: "never",
        // the pool queues and cancels requests; deck holds new ones back
        // while the view is still moving. A debounce turns on deck's
        // throttling, where maxRequests 0 would allow no request at all.
        maxRequests: 64,
        debounceTime: 60,
        maxCacheByteSize: 256 * 1024 * 1024,
        pickable,
        opacity,
        getTileData: (tile: any) => source.tile(tile.index.i, lod, tile.signal, urgent),
        renderSubLayers: (p: any) => {
          const t = p.data as PackedTile | null;
          if (!t || t.n === 0) return null;
          let data = datas.get(t);
          if (!data) {
            const attributes: Record<string, unknown> = { getPolygon: { value: t.positions, size: 2 } };
            if (t.indices) attributes.indices = { value: t.indices, size: 1 };
            data = { length: t.n, startIndices: t.starts, attributes };
            datas.set(t, data);
          }
          const swap = this.state.swap;
          return new CodesPolygonLayer({
            id: p.id,
            data: data as any,
            packed: t,
            palette,
            cover: base ? () => swap.baseCover : undefined,
            _normalize: false,
            _windingOrder: "CCW",
            positionFormat: "XY",
            pickable: p.pickable,
            opacity: p.opacity,
            extruded: false,
            stroked: false,
          } as any);
        },
      }),
    );
  }

  /**
   * The level to draw: the pixel target's, or the finest coarser one within
   * the budget. A level whose shards are still opening is taken as it is;
   * its layer reads nothing until they have, and the choice is made again
   * then (the source's change redraws).
   */
  private chooseLod(target: number, box: Box) {
    const { source, budget = DEFAULT_BUDGET } = this.props;
    const swap = this.state.swap;
    return chooseLevel(
      target,
      source.lods,
      (lod) => {
        void source.ensure(source.tierOf(lod), box);
        return source.costIn(lod, box);
      },
      budget,
      (lod) => lod === swap.shown || lod === swap.pending,
    );
  }

  renderLayers(): Layer[] {
    const { source, cellPx = 1 } = this.props;
    const swap = this.state.swap;
    const stats = this.state.stats;
    const vp = this.context.viewport;
    const box = viewportBox(vp);
    const target = pickLod(a5Resolution(vp, cellPx), source.lods);
    // the shards of the target's backdrop first, so a coarse view comes
    // before any judgement of the target's cost
    void source.ensure(source.tierOf(pickLod(target - BACKDROP_STEP, source.lods)), box);
    const { lod, capped, cost } = this.chooseLod(target, box);
    const tier = source.tierOf(lod);
    const back = pickLod(lod - BACKDROP_STEP, source.lods);
    swap.backdrop = back < lod ? back : null;
    // the backdrop's shards first: its footers come before the level's
    if (swap.backdrop != null) void source.ensure(source.tierOf(swap.backdrop), box);
    void source.ensure(tier, box);
    stats.targetLod = target;
    stats.capped = capped;
    stats.viewBytes = cost?.bytes ?? null;
    stats.viewCells = cost?.rows ?? null;

    const now = performance.now();
    let out: Layer[];
    if (swap.shown == null || lod === swap.shown) {
      swap.shown = lod;
      swap.pending = null;
      out = [this.tileLayer(lod)];
    } else {
      if (swap.pending !== lod) {
        swap.pending = lod;
        swap.pendingSince = now;
        swap.ready.delete(lod);
        swap.hold(MAX_HOLD_MS + 32);
      }
      if (swap.canShow(lod, now)) {
        swap.shown = lod;
        swap.pending = null;
        out = [this.tileLayer(lod)];
      } else {
        out = [this.tileLayer(swap.shown), this.tileLayer(lod)];
      }
    }
    // the backdrop's layer stays while it is one, drawn or not, so it is
    // ready for the next move; the level shown may be it (after a zoom in)
    const b = swap.backdrop;
    if (b != null && b !== swap.shown && b !== swap.pending) out.unshift(this.tileLayer(b, true));
    // the base level, always (kept loaded), at the bottom
    if (swap.base != null) out.unshift(this.tileLayer(swap.base, false, true));
    swap.updateCovers();
    stats.backdropLod = b;
    stats.backdrop = b != null ? (swap.contents.get(b) ?? []) : [];
    stats.backdropReady = b != null && !!swap.ready.get(b);
    const c = swap.loaded.get(swap.shown ?? lod) ?? { selected: 0, loaded: 0, cells: 0 };
    stats.lod = swap.shown;
    stats.ready = swap.shown != null && !!swap.ready.get(swap.shown);
    stats.tier = source.tierOf(swap.shown ?? lod);
    stats.pendingLod = swap.pending;
    stats.tilesSelected = c.selected;
    stats.tilesLoaded = c.loaded;
    stats.cellsDrawn = c.cells;
    stats.tiles = swap.contents.get(swap.shown ?? lod) ?? [];
    this.queueStats();
    return out;
  }
}
