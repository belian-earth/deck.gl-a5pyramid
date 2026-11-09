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
//     MAX_HOLD_MS, so the map neither blanks nor draws two levels at once.
// Different from a5view:
//   - the tiles of a LOD come from every open shard of its tier; shards open
//     when a view first meets them, and a new version of the source redoes
//     tile selection;
//   - getTileData returns the decode's promise, so deck.gl tracks loading
//     (a5view decoded outside deck and marked tiles dirty);
//   - cells finer than FAN_MIN_LOD come with fan triangles, so deck.gl skips
//     earcut, its largest per-tile cost in a5view;
//   - colours come from the stored codes on the main thread, so a new stretch
//     recolours tiles without reading them again.

import { CompositeLayer, type GetPickingInfoParams, type Layer, type LayerProps, type PickingInfo, type UpdateParameters } from "@deck.gl/core";
import { TileLayer, _Tileset2D as Tileset2D } from "@deck.gl/geo-layers";
import { SolidPolygonLayer } from "@deck.gl/layers";
import { a5Resolution, boxKey, pickLod, viewportBox } from "./tiling";
import { colours, type Stretch } from "./colour";
import type { PyramidSource } from "./source";
import type { PackedTile } from "./tile-core";

export const MAX_HOLD_MS = 1500;

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
}

export interface A5PyramidLayerProps extends LayerProps {
  source: PyramidSource;
  stretch: Stretch;
  /** Resolutions added to the zoom's A5 resolution; negative for coarser cells. */
  lodBias?: number;
  /** Filled in as the layer draws: the LOD shown and its tiles. Read it on a timer. */
  stats?: RenderStats;
}

export function emptyRenderStats(): RenderStats {
  return { lod: null, tier: 0, pendingLod: null, tilesSelected: 0, tilesLoaded: 0, cellsDrawn: 0 };
}

/** Shared by the parent and its TileLayers: which LOD is drawn. */
class Swap {
  shown: number | null = null;
  pending: number | null = null;
  pendingSince = 0;
  ready = new Map<number, boolean>();
  loaded = new Map<number, { selected: number; loaded: number; cells: number }>();
  redraw: () => void = () => {};
  private queued = false;

  draws(lod: number): boolean {
    if (this.pending == null) return lod === this.shown;
    return this.ready.get(this.pending) ? lod === this.pending : lod === this.shown;
  }

  sink: RenderStats | null = null;

  report(lod: number, ready: boolean, counts: { selected: number; loaded: number; cells: number }) {
    const was = this.ready.get(lod);
    this.ready.set(lod, ready);
    this.loaded.set(lod, counts);
    if (this.sink && lod === this.shown) {
      this.sink.tilesSelected = counts.selected;
      this.sink.tilesLoaded = counts.loaded;
      this.sink.cellsDrawn = counts.cells;
    }
    // the pending level just completed: let the parent drop the old one,
    // through a timer, not from inside deck's render pass
    if (lod === this.pending && ready && !was) this.later();
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
    for (const t of tiles) {
      if (t.isLoaded) {
        loaded++;
        cells += (t.content as PackedTile | null)?.n ?? 0;
      }
    }
    // complete once every tile selected has loaded and every shard of the
    // tier in view has opened (before that a level selects too few tiles)
    const { source, tier, lod, swap } = this.props;
    const opened = source.opened(tier, viewportBox(this.context.viewport));
    swap.report(lod, opened && this.isLoaded, { selected: tiles.length, loaded, cells });
    return out;
  }

  filterSubLayer(context: Parameters<TileLayer["filterSubLayer"]>[0]) {
    return this.props.swap.draws(this.props.lod) && super.filterSubLayer(context);
  }
}

function tilesetClass(source: PyramidSource, lod: number) {
  return class extends Tileset2D {
    private lastKey: string | null = null;
    private lastIndices: { i: string }[] = [];

    getTileIndices(opts: { viewport: any }) {
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
        values: t.values.map((v) => v[info.index]),
      } satisfies CellInfo;
    }
    return info;
  }

  private tileset(lod: number) {
    let cls = this.state.tilesets.get(lod);
    if (!cls) {
      cls = tilesetClass(this.props.source, lod);
      this.state.tilesets.set(lod, cls);
    }
    return cls;
  }

  private tileLayer(lod: number) {
    const { source, stretch, opacity, pickable } = this.props;
    const datas = this.state.datas;
    return new PyramidTileLayer(
      this.getSubLayerProps({
        id: `lod${lod}`,
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
        updateTriggers: { renderSubLayers: stretch.version },
        getTileData: (tile: any) => source.tile(tile.index.i, lod, tile.signal),
        renderSubLayers: (p: any) => {
          const t = p.data as PackedTile | null;
          if (!t || t.n === 0) return null;
          let d = datas.get(t);
          if (!d || d.version !== stretch.version) {
            const attributes: Record<string, unknown> = {
              getPolygon: { value: t.positions, size: 2 },
              getFillColor: { value: colours(t, stretch), size: 4 },
            };
            if (t.indices) attributes.indices = { value: t.indices, size: 1 };
            d = { version: stretch.version, data: { length: t.n, startIndices: t.starts, attributes } };
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
    void source.ensure(tier, viewportBox(vp));

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
      if (swap.ready.get(lod) || now - swap.pendingSince >= MAX_HOLD_MS) {
        swap.shown = lod;
        swap.pending = null;
        out = [this.tileLayer(lod)];
      } else {
        out = [this.tileLayer(swap.shown), this.tileLayer(lod)];
      }
    }
    if (stats) {
      const c = swap.loaded.get(swap.shown ?? lod) ?? { selected: 0, loaded: 0, cells: 0 };
      stats.lod = swap.shown;
      stats.tier = source.tierOf(swap.shown ?? lod);
      stats.pendingLod = swap.pending;
      stats.tilesSelected = c.selected;
      stats.tilesLoaded = c.loaded;
      stats.cellsDrawn = c.cells;
    }
    return out;
  }
}
