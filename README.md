# deck.gl-a5pyramid

Draw [a5pyramid](https://github.com/belian-earth/a5pyramid) sets in
[deck.gl](https://deck.gl): pentagonal [A5](https://github.com/felixpalmer/a5) cells at many
levels of detail, stored as Parquet and read straight from static storage
(S3, a CDN, any server with byte ranges) with HTTP range requests. No tile
server.

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
- `A5PyramidLayer` props: `source`, `colouring`, `cellPx` (default 1),
  `budget` (`{ bytes, cells }`, default `DEFAULT_BUDGET`, 40 MB and 1.5 M
  cells), `onStats(stats)` (the levels drawn, their tiles, the view's cost),
  and deck.gl's `pickable` and `opacity`. Picking gives a `CellInfo`: the
  cell id, its LOD, `ncell` and its codes.
- Colourings: `bandColouring(stretch)`, `neutralColouring()`,
  `pcaColouring(basis)`, or `colouring(offset, decode, fill)` for any affine
  map of codes or decoded values.
- `ScreenPca`, `pcaBasis`, `scores`: screen PCA.
- `a5Resolution`, `cellPixels`, `chooseLevel` and friends: the level rules,
  for overlays and tests.

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

## Licence

Apache 2.0. The AlphaEarth Foundations Satellite Embedding dataset is produced
by Google and Google DeepMind and published under CC BY 4.0.
