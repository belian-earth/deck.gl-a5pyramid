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
//   - colours come from the stored codes on the main thread through a
//     Colouring (a band stretch, or the PCA of the view), so new colours
//     apply to tiles without reading them again;
//   - a backdrop level, BACKDROP_STEP LODs coarser than the one wanted (a
//     sixteenth of its cells at 2), is read ahead of every other tile and
//     drawn under a level still loading, so a view is covered, coarsely,
//     after one or two small reads.

import { CompositeLayer, type GetPickingInfoParams, type Layer, type LayerProps, type PickingInfo, type UpdateParameters } from "@deck.gl/core";
import { TileLayer, _Tileset2D as Tileset2D } from "@deck.gl/geo-layers";
import { SolidPolygonLayer } from "@deck.gl/layers";
import { a5Resolution, boxKey, pickLod, viewportBox } from "./tiling";
import type { Colouring } from "./colour";
import type { PyramidSource } from "./source";
import type { PackedTile } from "./tile-core";

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
  /** Resolutions added to the zoom's A5 resolution; negative for coarser cells. */
  lodBias?: number;
  /** Filled in as the layer draws: the LOD shown and its tiles. Read it on a timer. */
  stats?: RenderStats;
}

export function emptyRenderStats(): RenderStats {
  return {
    lod: null, tier: 0, pendingLod: null, tilesSelected: 0, tilesLoaded: 0, cellsDrawn: 0, tiles: [],
    backdropLod: null, backdrop: [], backdropReady: false,
  };
}

/** Shared by the parent and its TileLayers: which LOD is drawn. */
class Swap {
  shown: number | null = null;
  pending: number | null = null;
  backdrop: number | null = null;
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

  sink: RenderStats | null = null;

  report(lod: number, ready: boolean, counts: { selected: number; loaded: number; cells: number }, contents: PackedTile[]) {
    const was = this.ready.get(lod);
    const before = this.loaded.get(lod)?.loaded;
    this.ready.set(lod, ready);
    this.loaded.set(lod, counts);
    this.contents.set(lod, contents);
    if (this.sink && lod === this.shown) {
      this.sink.tilesSelected = counts.selected;
      this.sink.tilesLoaded = counts.loaded;
      this.sink.cellsDrawn = counts.cells;
      this.sink.tiles = contents;
    }
    if (this.sink && lod === this.backdrop) {
      this.sink.backdrop = contents;
      this.sink.backdropReady = ready;
    }
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
    const { source, tier, lod, swap } = this.props;
    const opened = source.opened(tier, viewportBox(this.context.viewport));
    // (the tiles' own flags: the layer's isLoaded also waits for sublayers
    // made in this very pass, so a level's last tile would never count)
    swap.report(lod, opened && !!this.state.tileset && loaded === tiles.length, { selected: tiles.length, loaded, cells }, contents);
    return out;
  }

  filterSubLayer(context: Parameters<TileLayer["filterSubLayer"]>[0]) {
    return this.props.swap.draws(this.props.lod) && super.filterSubLayer(context);
  }
}

function tilesetClass(source: PyramidSource, lod: number, swap: Swap) {
  return class extends Tileset2D {
    private lastKey: string | null = null;
    private lastIndices: { i: string }[] = [];

    getTileIndices(opts: { viewport: any }) {
      // a level held on screen while another loads keeps the tiles it has:
      // it would otherwise read tiles for a view it will not be shown in
      // (after a zoom out, a hundred of them)
      if (this.lastKey != null && swap.pending != null && lod !== swap.pending && lod !== swap.backdrop) {
        return this.lastIndices as any;
      }
      const box = viewportBox(opts.viewport);
      const key = boxKey(box) + "|" + source.version;
      if (key !== this.lastKey) {
        this.lastKey = key;
        this.lastIndices = source.tilesIn(lod, box).map((t) => ({ i: t.key }));
      }
      return this.lastIndices as any;
    }
    getTileId(index: any) {
      return index ? index.i : null;
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
  static defaultProps = { lodBias: 0, pickable: true };

  declare state: {
    swap: Swap;
    tilesets: Map<number, ReturnType<typeof tilesetClass>>;
    datas: WeakMap<PackedTile, { version: number; data: unknown }>;
  };

  initializeState() {
    const swap = new Swap();
    swap.redraw = () => this.setNeedsUpdate();
    this.setState({ swap, tilesets: new Map(), datas: new WeakMap() });
    this.props.source.onChange = () => swap.later();
  }

  shouldUpdateState({ changeFlags }: UpdateParameters<this>) {
    return changeFlags.somethingChanged;
  }

  updateState({ props, oldProps }: UpdateParameters<this>) {
    if (props.source !== oldProps.source) {
      this.state.tilesets.clear();
      this.state.swap.shown = null;
      this.state.swap.pending = null;
      this.state.swap.backdrop = null;
      this.state.swap.contents.clear();
      this.state.swap.ready.clear();
      this.state.swap.loaded.clear();
      props.source.onChange = () => this.state.swap.later();
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

  private tileLayer(lod: number, urgent = false) {
    const { source, colouring, opacity, pickable } = this.props;
    const datas = this.state.datas;
    return new PyramidTileLayer(
      this.getSubLayerProps({
        // per source: deck keeps a TileLayer's tileset across prop changes,
        // so a new source needs new layers, not new props
        id: `s${source.id}-lod${lod}`,
        lod,
        tier: source.tierOf(lod),
        source,
        swap: this.state.swap,
        version: source.version,
        TilesetClass: this.tileset(lod),
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
        updateTriggers: { renderSubLayers: colouring.version },
        getTileData: (tile: any) => source.tile(tile.index.i, lod, tile.signal, urgent),
        renderSubLayers: (p: any) => {
          const t = p.data as PackedTile | null;
          if (!t || t.n === 0) return null;
          let d = datas.get(t);
          if (!d || d.version !== colouring.version) {
            const attributes: Record<string, unknown> = {
              getPolygon: { value: t.positions, size: 2 },
              getFillColor: { value: colouring.colour(t), size: 4 },
            };
            if (t.indices) attributes.indices = { value: t.indices, size: 1 };
            d = { version: colouring.version, data: { length: t.n, startIndices: t.starts, attributes } };
            datas.set(t, d);
          }
          return new SolidPolygonLayer({
            id: p.id,
            data: d.data as any,
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

  renderLayers(): Layer[] {
    const { source, lodBias = 0, stats } = this.props;
    const swap = this.state.swap;
    swap.sink = stats ?? null;
    const vp = this.context.viewport;
    const lod = pickLod(a5Resolution(vp, lodBias), source.lods);
    const tier = source.tierOf(lod);
    const box = viewportBox(vp);
    const back = pickLod(lod - BACKDROP_STEP, source.lods);
    swap.backdrop = back < lod ? back : null;
    // the backdrop's shards first: its footers come before the level's
    if (swap.backdrop != null) void source.ensure(source.tierOf(swap.backdrop), box);
    void source.ensure(tier, box);

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
    if (stats) {
      stats.backdropLod = b;
      stats.backdrop = b != null ? (swap.contents.get(b) ?? []) : [];
      stats.backdropReady = b != null && !!swap.ready.get(b);
      const c = swap.loaded.get(swap.shown ?? lod) ?? { selected: 0, loaded: 0, cells: 0 };
      stats.lod = swap.shown;
      stats.tier = source.tierOf(swap.shown ?? lod);
      stats.pendingLod = swap.pending;
      stats.tilesSelected = c.selected;
      stats.tilesLoaded = c.loaded;
      stats.cellsDrawn = c.cells;
      stats.tiles = swap.contents.get(swap.shown ?? lod) ?? [];
    }
    return out;
  }
}
