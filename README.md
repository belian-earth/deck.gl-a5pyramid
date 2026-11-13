# deck.gl-a5pyramid

A [deck.gl](https://deck.gl) layer that streams
[A5](https://github.com/felixpalmer/a5) pyramids straight from Parquet: it
reads only the tiles a view needs, at the detail the screen can show, and
colours them on the GPU. No tile server: an
[a5pyramid](https://github.com/belian-earth/a5pyramid) set on static storage
(S3, a CDN, any server with byte ranges) is read with HTTP range requests.
It is to A5 pyramids roughly what
[deck.gl-raster](https://github.com/developmentseed/deck.gl-raster) is to
Cloud-Optimized GeoTIFFs.

- **Coloured on the GPU.** Each tile's codes go to the GPU once; a colouring
  (a band stretch, or the principal components of the cells in view) is a
  few uniforms, so recolouring costs no pass over the cells.
- **Detail by pixels, bounded by a budget.** The level drawn has cells of at
  most `cellPx` CSS pixels, unless that view would read more than a budget
  of bytes or cells (known exactly from the Parquet footers), when it steps
  coarser.
- **Never blank.** A coarse backdrop under a level still loading, and a base
  level of the whole set kept from the start, so pans and jumps show
  something at once; the base is masked where finer data has arrived, so its
  large cells do not stand out past a coast.
- **Read once.** Tiles decode in workers, nearest the view's centre first;
  tiles that leave the view are aborted; decoded tiles are cached.

Built for the AlphaEarth Foundations satellite embeddings (64 int8 bands a
cell) at Belian; any a5pyramid set with int8 columns or an int8 vector column
draws the same way. How it works, and why: [`docs/design.md`](docs/design.md).

**Status:** 0.1, on GitHub only; the API may change before a release on npm.
It requires deck.gl and luma.gl 9.4 (it relies on parts of deck.gl that are
not public API; see the design notes).

## Install

```sh
npm install github:belian-earth/deck.gl-a5pyramid
```

with `@deck.gl/core`, `@deck.gl/layers`, `@deck.gl/geo-layers`,
`@luma.gl/core`, `@luma.gl/engine` and `@luma.gl/shadertools` at 9.4.x. The
decoding runs in a module worker the package loads with
`new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`,
which Vite, webpack 5 and Rollup bundle as they find it; pass
`worker: () => new Worker(...)` to `PyramidSource.open` to host it yourself.

## Use

```ts
import { Deck } from "@deck.gl/core";
import { A5PyramidLayer, PyramidSource, bandColouring } from "deck.gl-a5pyramid";

const source = await PyramidSource.open("https://data.example.org/sabah/aef-a09-a11/year=2025/");

new Deck({
  initialViewState: { longitude: 117.3, latitude: 5.4, zoom: 7 },
  controller: true,
  layers: [
    new A5PyramidLayer({
      id: "aef",
      source,
      // the first three columns as red, green and blue, each stretched between two codes
      colouring: bandColouring({ ranges: [[-43, 69], [-60, 34], [22, 72]], version: 1 }),
      cellPx: 1,
      pickable: true,
    }),
  ],
});
```

Over MapLibre or Mapbox, add the layer to a `MapboxOverlay` from
`@deck.gl/mapbox` as any deck.gl layer.

### Screen PCA

For a vector set (as 64-band embeddings), colour by the view's first three
principal components, recomputed as the view changes:

```ts
import { Deck } from "@deck.gl/core";
import { A5PyramidLayer, PyramidSource, ScreenPca, isSettled, neutralColouring } from "deck.gl-a5pyramid";

const source = await PyramidSource.open("https://data.example.org/sabah/aef-64/year=2025/");

const pca = new ScreenPca();
let colouring = neutralColouring();

function layer(): A5PyramidLayer {
  return new A5PyramidLayer({
    id: "aef", source, colouring, cellPx: 2,
    onStats: (stats) => {
      const next = pca.update(stats, isSettled(stats, source));
      if (next) {
        colouring = next;
        deck.setProps({ layers: [layer()] });
      }
    },
  });
}

const deck = new Deck({ initialViewState: { longitude: 117.8, latitude: 4.96, zoom: 10 }, controller: true, layers: [layer()] });
```

### API

- `PyramidSource.open(url, { workers, cacheBytes, worker })`: a set, from the
  URL of its directory (holding `set.json`). `source.close()` stops its
  workers.
- `PyramidSource.fromSet(set, { refresh, pca, ... })`: a set given whole,
  its shards with `url`s (signed by a service, say); `refresh` fetches it
  again when a read is refused (403, 404, 410). `pca: true` reads the PCA
  sums `ScreenPca` needs for a set whose metadata does not ask for them.
- `region` (either constructor): draw only the cells of a region, given as
  polygons' outer rings (`[[lon, lat], ...]` each, holes filled, not
  crossing the antimeridian). See "A region" below.
- `merge`: tiles of one row group that a view asks for together are read
  with one request a column. Off by default for http(s) URLs, where it cut
  the requests in flight and slowed reads over a CDN; on for local files.
- `A5PyramidLayer` props: `source`, `colouring`, `cellPx` (default 1),
  `budget` (`{ bytes, cells }`, default `DEFAULT_BUDGET`, 40 MB and 1.5 M
  cells), `onStats(stats)` (the levels drawn, their tiles, the view's cost),
  and deck.gl's `pickable`, `opacity` and `extensions` (with their props:
  a `MaskExtension` and `maskId` clip the cells to a mask layer; a source's
  `region` draws whole cells instead).
  Picking gives a `CellInfo`: the
  cell id, its LOD, `ncell` and its codes.
- Colourings: `bandColouring(stretch)`, `neutralColouring()`,
  `pcaColouring(basis)`, or `colouring(offset, decode, fill)` for any affine
  map of codes or decoded values.
- `ScreenPca`, `pcaBasis`, `scores`: screen PCA.
- `a5Resolution`, `cellPixels`, `chooseLevel` and friends: the level rules,
  for overlays and tests.

### A region

```ts
const source = PyramidSource.fromSet(set, { region: rings, pca: true });
```

Each level's cells are those of a5-js `polygonToCells(ring, level, {
containment: "overlapping" })`, each ring's covering unioned: every cell
that touches a ring is drawn, whole, and no other. The coverings are built
in the decode workers when a level is first read there (on a region of 19
rings: 3 to 32 ms a level up to 14, 62 ms at 15, about 0.5 s at 18) and
kept; a tile's rows outside them are dropped before packing and before
the PCA sums, so screen PCA describes the region alone.

What cannot hold a cell of the region is not read: a shard whose box meets
no ring's box (not even its footer), and a tile whose cell id bounds miss
its level's covering. The bounds come from the footer already fetched: the
cell column's statistics where a tile is a row group, the column index
where a tile is a page of a larger row group; a file with neither falls
back to the tile's box against the rings' boxes. `source.stats.tilesSkipped`
and `rowsDropped` count them.

## Data

Write sets with a5pyramid (R, with a Rust core): `a5_write_pyramid_set()`.
Serve the directory as static files with byte ranges and CORS allowing the
`Range` header (and exposing `Content-Range`).

## Develop

```sh
npm install
npm run check          # types
npm test               # unit tests, Node, against test/fixtures
npm run test:browser   # GPU colours read back in the installed Chrome
npm run build          # dist/
```

`test/fixtures` holds two small sets written by a5pyramid
(`tools/fixtures.R`): three int8 bands, and a 64-code vector.

## Acknowledgements

- **[deck.gl-raster](https://github.com/developmentseed/deck.gl-raster)**
  (Development Seed), read at commit 6da743d for what makes it fast. Its
  approaches, not its code, are behind: colouring on the GPU from codes
  uploaded once, with the colouring as shader uniforms (its AEF mosaic
  example draws the same embeddings); aborting the loads of tiles that left
  the view at once (`abortUnwantedLoads`); choosing the level from screen
  pixels; and leaving range requests uncoalesced, after its finding that
  coalescing gains nothing at about 1 MB a tile.
- **[A5](https://github.com/felixpalmer/a5)** (Felix Palmer), through
  a5-js. Its cells are equal-area, which makes the pixel rule exact, and its
  ids follow a space-filling curve, so a tile, a run of consecutive ids, is
  a compact patch, which the base level's mask relies on.
- **[hyparquet](https://github.com/hyparam/hyparquet)** (Hyperparam):
  Parquet read in the browser, a row group at a time, over range requests.
- **[deck.gl](https://deck.gl) and [luma.gl](https://luma.gl)** (vis.gl,
  OpenJS Foundation): the rendering, including the internals this layer
  depends on.
- **AlphaEarth Foundations Satellite Embedding dataset** (Google and Google
  DeepMind, CC BY 4.0): the 64-band embeddings this was built to draw. The
  test fixtures are synthetic.

## Licence

Apache 2.0.
