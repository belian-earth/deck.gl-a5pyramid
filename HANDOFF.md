# Handoff: deck.gl-a5pyramid

Updated 2026-11-12. Design: `docs/design.md`.

## State

- Split from the a5-pyramids demo (`~/belian/a5-pyramid-demo`,
  `web/src/lib/pyramid/`) on 2026-11-10 with `git subtree split`, so the
  renderer's commits came along. Public at
  https://github.com/belian-earth/deck.gl-a5pyramid since 2026-11-10; the
  demo installs `github:belian-earth/deck.gl-a5pyramid#v0.1.0`.
  GitHub only for now; npm once the API settles (Hugh, 2026-11-10). Licence
  Apache 2.0, as a5pyramid.
- `npm run check`, `npm test` (28 unit tests in Node against two fixture
  sets written by a5pyramid) and `npm run test:browser` (GPU colours read
  back in headless Chrome: 134 cells a set, within 1/255 of the formula)
  pass. CI runs all three and the build; the first run on GitHub passed
  with the same browser figures (134 cells a set, worst 1 and 0).
- Branch `flat-list-vectors` (2026-11-12): a vector stored as a
  fixed-length list (LIST<INT8>, store schema v4, belian-production
  `docs/store-v4.md`) is read by `src/fixed-list.ts`, not `parquetRead`.
  - It walks the chunk's pages with hyparquet's primitives and writes the
    codes straight into the tile's codes texture. Rows are checked whole
    (64 elements, no nulls) from a v2 page header or from the level runs,
    and the levels are decoded only when their runs take another shape.
  - On a 16,384-row v4 leaf tile in Node: `parquetRead` 44 to 54 ms, this
    path 6.7 ms, fixed binary 5.5 ms (zstd by zstddec, wasm). fzstd, the
    pure-JS zstd of hyparquet-compressors, adds about 20 ms a tile.
  - A whole v4 leaf tile: 3 ms read, about 6 ms vector, 17 ms PCA sums,
    47 ms A5 boundaries.
  - `ncell` is optional (v4 leaves carry none; each counts 1).
  - `setCompressors()` takes decompressors for the worker (zstd for v4).
  - Test fixture `vector-list` (`tools/fixture-vector-list.R`): the vector
    fixture as a list, read tile for tile identically.
- Peer dependencies are pinned to deck.gl and luma.gl 9.4.x: the layer uses
  deck.gl internals (listed in the design notes). Test a new minor before
  widening the range.

## Next

0. Store v4 (belian-production PR #277): the worker sets a wasm zstd
   (zstddec measured best) through `setCompressors()`; the source over a
   job's set; the format v4 gate; the drawing rule for partial rows.
1. a5view could load this package instead of its own copy of the lazy
   renderer, one renderer for both.
2. Propose to deck.gl: abort loading tiles that left the view whatever the
   number in flight (here an override of `Tileset2D.update`).
3. From the design notes, not yet taken: fetching apart from decoding,
   instanced pentagons, a tile-size sweep for 64-byte rows.
