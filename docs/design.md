# Design

How deck.gl-a5pyramid reads and draws an a5pyramid set, and why. The set
format is a5pyramid's
[`FORMAT.md`](https://github.com/belian-earth/a5pyramid/blob/main/src/rust/a5pyramid/FORMAT.md);
this reader takes format versions 1 to 3 (3 adds vector columns).

## The data

A set is a directory: `set.json` lists its tiers and shards, and each shard
is a Parquet file holding several levels of detail (LODs) of A5 cells for one
coarse cell. A tier is a range of LODs at one shard resolution: by default
LODs 2 to 12 in shards of resolution 0, 13 to 15 in shards of resolution 3,
16 to 18 in shards of resolution 6. Within a shard each level is cut into
tiles, one Parquet row group each, of at most `tile_size` rows. A tile is a
run of consecutive cell ids, so a compact patch along A5's space-filling
curve, and its box is in the shard's index (the `a5pyramid` key-value entry
of the footer). A row is a cell: its id, `ncell` (the leaves it stands for)
and its values, int8 codes (scalar columns, or one vector column of `dim`
codes per cell, as AEF's 64-dimensional embeddings).

## Reading

- **set.json, then footers.** Opening a set reads `set.json`. A shard's
  footer is read with one range request (`set.json` gives its size), parsed
  in a worker, and kept; the worker also returns each tile's rows and
  compressed bytes, for the view budget. Footers are read only for shards a
  view meets.
- **Tiles in workers.** A tile is read and decoded in a pool of workers
  (hyparquet over range requests; hyparquet merges a row group's column
  chunks), then packed for deck.gl: A5 boundaries from a5-js (open rings),
  fan triangles for cells at LOD 9 and finer (convex in lon/lat, so deck.gl
  skips earcut, its largest cost per tile), the codes laid out as a texture
  (below), and, for PCA, the tile's weighted sums of its decoded values.
  Packing is mostly `cellToBoundary`, about 3.5 us a cell.
- **Order.** Each request joins a queue. Urgent requests (shard indexes and
  the backdrop's tiles) go first, and nothing else starts while one runs, so
  on a slow link they have the bandwidth. Otherwise the newest batch (one
  task's requests) is served first, in the order deck.gl asked, which is
  nearest the view's centre first.
- **Aborts.** deck.gl aborts a loading tile that left the view only when
  more than `maxRequests` are loading; each level's tileset aborts such
  tiles at once (deck.gl-raster's `abortUnwantedLoads`), and the queue drops
  a job no one waits for any more.
- **Cache.** Decoded tiles are kept, least recently used dropped first,
  up to `cacheBytes` (512 MB by default), so nothing is read twice while it
  fits.

## Choosing a level

- **The pixel rule.** A5 cells at a resolution have equal areas, so a
  cell's size on screen follows from the resolution and the view's scale
  (Web Mercator at 512-pixel tiles, at the view's centre). The level drawn
  is the coarsest whose cells are at most `cellPx` CSS pixels across (the
  square root of their area); CSS rather than device pixels, so a
  high-density screen does not quadruple the bytes. a5view's table of
  (zoom, resolution) anchors, which this replaced, gave 0.6 to 3.3 px a cell
  depending on the zoom.
- **The view budget.** A view full of data at middle zooms holds many cells:
  at 2 px, about a million for a laptop screen, 80 MB of 64-band
  embeddings. So the level is the pixel rule's unless the tiles it would
  read for the view exceed a budget (40 MB compressed or 1.5 million cells
  by default); then it is the finest coarser level within both. The cost is
  exact, from the footers: the rows and bytes of the tiles meeting the view,
  which are what is fetched. A view sparse in data (a region's overview,
  mostly sea) is drawn at the pixel rule's level; a full one is capped. A
  level selects no tiles until every shard of its tier in view has opened,
  so a level the budget turns down never starts reading, and a level on
  screen keeps a quarter over the budget before it gives way, so a view near
  the limit does not flip between levels as it pans. `chooseLevel` is the
  rule, a pure function.

## Changing levels

Each level is drawn through its own deck.gl TileLayer, whose tileset holds
the level's tiles as atomic tiles (A5 is not the quadtree deck.gl assumes:
no parents, no refinement).

- **Swap.** On a level change the new level loads undrawn and replaces the
  old one in the first frame that all its tiles in view are in, or after
  1.5 s once half of them are, so the map neither blanks nor draws two
  levels at once. A level held on screen meanwhile adds no tiles (after a
  zoom out it would read a hundred for a view it will not be shown in) and
  drops, so aborts, those that left the view.
- **Backdrop.** The level two coarser than the one wanted (a sixteenth of its
  cells) is read before any other tile and drawn under a level still
  loading, so a view is covered, coarsely, after one or two small reads.
- **Base.** The finest level of tier 0 whose tiles over the whole set fit
  150 k cells and 8 MB (for a region such as Sabah, LOD 12, three tiles) is
  read once, kept and drawn under the rest while they load, so a jump
  anywhere in the set shows something at once.
- **Base mask.** Drawn whole, the base's large cells stand out past a coast,
  over sea where finer levels have no cells. A base cell is not drawn when
  the finer cell at its centre lies in a loaded tile's run of a level drawn
  above: a tile's run includes the sea between its cells, so there the finer
  level has its data or knows there is none. A mask by the tiles' boxes
  instead left gaps while panning, as a box reaches over neighbouring tiles
  not yet loaded. The finer cell is an a5-js lookup (about 3.5 us), made
  only for base cells in view, kept per level, at most 1,500 a frame; a cell
  not yet looked up stays drawn.

## Drawing and colouring

- **Polygons.** Each tile is one SolidPolygonLayer over binary attributes
  (positions, fan indices), never PolygonLayer, which draws nothing from
  binary data.
- **Colour on the GPU** (as deck.gl-raster colours its rasters). A tile's
  codes go to the GPU once, as an RGBA8 texture: four codes a texel, a
  cell's texels in one row (a 64-code vector is read from Parquet straight
  into the texture's buffer). A colouring is an affine map from a cell's
  codes, or the AEF values they decode to (sign(x) (x / 127.5)^2), to red,
  green and blue, clamped to 0..1: 48 vec4 uniforms of weights and an
  offset. The vertex shader finds its cell from deck.gl's per-vertex row
  index (the `rowIndexes` attribute picking already uses), reads the
  codes and dots them, four at a time, with each channel's weights, and sets
  `vColor` at `vs:#main-end`. A new colouring is a uniform update and a
  redraw: no pass over cells, no new attributes. At 591 k cells (3.1 M
  vertices) on an Intel Iris Xe the colouring costs about 0.3 ms a frame.
- **Colourings.** `bandColouring` (three columns as red, green and blue,
  each stretched between two codes), `neutralColouring` (one grey) and
  `pcaColouring`. Any affine map of codes or values fits the same uniforms.

## Screen PCA

For a vector set, colours can follow the view: the first three principal
components of the cells in view, each stretched over the view's mean
+/- 2 standard deviations. Each tile arrives with sums of its decoded values
(weighted by `ncell`, from at most 4,096 of its rows), so the covariance of
the view is a sum over tiles, solved (cyclic Jacobi, 64 x 64) in a few
milliseconds. A new basis is matched to the previous one (axis order and
sign), so colours shift with the view rather than jump. `ScreenPca` decides
when: while a view loads, the basis comes from the complete backdrop, or
from whichever level has tiles in, redone as their number doubles; once the
view settles, from every tile of the level shown.

## What deck.gl must keep

The layer relies on parts of deck.gl that are not public API, so it pins
deck.gl and luma.gl to 9.4.x:

- `_Tileset2D` and its `_cache`, `tile.abort()`, `getParentIndex` and
  `_getNearestAncestor` (overridden to drop the quadtree);
- SolidPolygonLayer's `rowIndexes` attribute, its `vColor` varying, its
  `topModel`, and the `vs:#main-end` injection point;
- luma.gl 9.4's uniform arrays (`["vec4<f32>", 48]`) and
  `Texture.writeData`.

## Measurements (2026-11-10, London, laptop with Intel Iris Xe)

- **CloudFront.** One 1 MB range: 0.12 to 0.16 s from the edge (first byte
  32 to 50 ms), 1.3 to 2.8 s from S3 in us-west-2 directly.
- **Sabah, 64 bands, overview** (LOD 13, 149 k cells, 9.6 MB): coarse colours
  after 2 to 3 s and complete after 6 to 10 s with cold caches; 1.3 s from
  the browser's cache.
- **Tours** (eight camera moves, Sabah to Danum Valley's leaves to Kinabalu):
  3-band set 31 MB in all, 64-band set 202 MB (358 MB at 2 px without the
  budget), no task over 120 ms on the main thread.

## Lessons from deck.gl-raster

Development Seed's raster layers for deck.gl, read for what makes them fast
(their `examples/aef-mosaic` draws the same AEF embeddings): colour on the
GPU from codes uploaded once (done here), serve the view's centre first and
abort what left it (done), choose the level from pixels (done). Not yet
taken: fetching on the main thread apart from decoding in workers, and
instanced pentagons (every A5 cell at resolution 6 and finer has five
vertices, so a cell could be five offsets and a fan, about 40 bytes instead
of 136, skipping deck.gl's tessellator). Range coalescing gains nothing at
about 1 MB a tile.
