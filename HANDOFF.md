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
- `npm run check`, `npm test` (41 unit tests in Node against fixture sets
  written by a5pyramid) and `npm run test:browser` (GPU colours read
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
- Same branch, later on 2026-11-12: tiles read in pieces, measured one
  option at a time on Tiisi's nine store partitions in 2021 (the browser
  tour's eight views through the source and worker in Node, warm second
  pass; MB, requests, cells packed, worker s):
  - baseline 91.8, 258, 2.16 M, 7.8; PCA 1 row in 16: 7.1 s; one range a
    tile: 111 requests. Both are the defaults since b8a0c07.
  - Probed pieces (tiles cut by their cells): 94.5, 368, 1.45 M, 7.3. Not
    adopted, and removed.
  - Page pieces of 2,048 rows: 73.3 MB but 1,767 requests. Not adopted.
  - Files written by parquet-rs were 12% larger than DuckDB's: v2 data
    pages and a dictionary tried on `cell`. a5pyramid now writes v1 pages.
- Same branch, 2026-11-12, after Hugh's row-group decision: **format version
  4** (a5pyramid#6). A row group of up to 131,072 rows holds tiles of 16,384
  rows, each one data page of every column, with the page index; ramet
  reads whole row groups, the renderer reads tiles.
  - The decision's figures on Tiisi: 16,384-row groups 92.3 MB, 111
    requests, 7.0 s; 131,072-row groups with 16,384-row pages read as tiles
    90.9, 258, 6.8; read whole 213.2, 43, 15.1. The ramet replay over S3 was
    27 to 41% slower with 16,384-row groups.
  - **The worker** maps an index tile to its row group and page
    (`locateTiles()`: row group `g` of `n` rows holds `ceil(n / tile_size)`
    tiles where the index has `row_group_size`), reads a whole row group
    for a tile that is one, and otherwise the tile's data page of each
    column (one request a column) with the row group's dictionary pages.
    The page index comes with a set's footer (`footer_bytes` covers it),
    so it costs no request.
  - **The source** treats the index's tiles as tiles; tiles of one row group
    share a worker (`affinity`), which holds its page index and
    dictionaries. `PieceOptions` became `ReadTuning` (`read.singleRange`,
    `read.statsEvery`); the prototype's `piece_rows` metadata and probed
    pieces are gone. The format gate is version 4.
  - **Tests:** `test/row-groups.test.ts` against the fixture `vector-rg`
    (`tools/fixture-vector-rg.R`, a5pyramid#6: tiles of 16 in row groups of
    64): every level's cells and codes equal the one-tile-a-row-group set's,
    and a tile reads by its pages with no request past the column chunks.
    The browser test draws `vector-rg` too (134 cells from 69 tiles, worst
    0). `test/tour-bench.test.ts` (was `pieces-bench`, `TOUR_BENCH`)
    compares sets; its Tiisi sets were deleted and need rebuilding.
- Same branch, 2026-11-12, the rest of format version 4 (a5pyramid#6,
  9bebc07) and the store's renderer items (belian-production
  `docs/store-v4.md` 6.2 and 9):
  - **Sets:** `PyramidSource.open()` reads `set.json` or `set.parquet` (a
    shard table, `readSetTable()`; a directory without `set.json` falls
    back to it, whether the server answers 403, 404 or a page);
    `PyramidSource.fromSet()` takes a set given whole, each shard with a
    `url` (a signed URL from belianapi) or a `path`. Empty shards (no
    file) are skipped; unwrapped boxes are wrapped.
  - **Refresh:** with `refresh`, a footer or tile read refused with HTTP
    403, 404 or 410 fetches the set once more and is retried; a shard
    whose file changed is reopened and its tiles dropped.
  - **Partial rows** are all drawn (every row, the rule of 3.3); the
    weights column is not read; list vectors, `ncell` and `lod` optional
    as before.
  - **Base level:** a tier of one level (a store's) is costed from the
    set's rows without a footer, and the base level goes up through such
    tiers while they fit the budget.
  - **zstd:** the bundled worker decodes Zstandard pages with zstddec
    (0.2.0, a dependency now), set up before its first message.
  - **Merged reads:** the pool dispatches in a microtask, and a decode job
    takes up to 7 more queued tiles of its row group (`Pool.MERGE`); the
    worker reads each column once for each run of consecutive pages
    (`readTiles()`). `merge: false` turns it off.
  - **Tests:** `test/store-v4.test.ts` on the fixture `store-v4`
    (`tools/fixture-store-v4.R`): a store in miniature, ten single-level
    tiers, three partitions with partial rows and `owner`, list vectors,
    `n_pixels`, an empty partition, `set.parquet` only. Every partial row
    drawn and every leaf counted once; merged reads take under half the
    requests; the base level from the set's rows without a footer; signed
    URLs refreshed once on 403. The browser test draws it (134 cells from
    70 tiles, worst 0). 41 unit tests.
  - **Danum, a whole store partition** (79f6800000000000, 2025, 16.8 M
    leaves from the a5-pyramid-demo extract, `n_pixels` set to 5), written
    by a5pyramid#6 as a store set two ways, levels 7 to 18 a file each:
    leaves 892.7 MB in 131,072-row groups (footer and page index 141 KB)
    against 894.0 MB in 16,384-row groups (footer 290 KB). The tour
    (`test/tour-danum.json`, eight views; MB, requests, wall s):
    - 16,384-row groups: 90.2, 107, 4.6.
    - 131,072-row groups, tiles read one at a time: 91.4, 260, 4.5.
    - 131,072-row groups, merged reads (default): 91.4, 89, 4.6.
    Wall time is the same within 3%; the worker's read time rises with
    merging in this harness only because its workers share one thread.
    `TOUR_BENCH=1 TOUR_DATA=<dir> TOUR_SETS=danum-16k,danum-v4
    TOUR_STEPS=test/tour-danum.json npx vitest run test/tour-bench.test.ts`.
- Peer dependencies are pinned to deck.gl and luma.gl 9.4.x: the layer uses
  deck.gl internals (listed in the design notes). Test a new minor before
  widening the range.

## Next

0. Store v4 is in (PR #4, merged 2026-11-12). Layer extensions reach the
   cell layers (branch `layer-extensions`): the ramet47 app masks a job's
   embeddings to its donor region with a `MaskExtension`; a browser test
   checks nothing is drawn outside the mask (230 sampled pixels drawn there
   without the change, 0 with it). A tag `v0.2.0` after it merges lets
   apps pin a release rather than a commit (Hugh's).
1. a5view could load this package instead of its own copy of the lazy
   renderer, one renderer for both.
2. Propose to deck.gl: abort loading tiles that left the view whatever the
   number in flight (here an override of `Tileset2D.update`).
3. From the design notes, not yet taken: fetching apart from decoding,
   instanced pentagons, a tile-size sweep for 64-byte rows.
