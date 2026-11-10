# Handoff: deck.gl-a5pyramid

Updated 2026-11-10. Design: `docs/design.md`.

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
- Peer dependencies are pinned to deck.gl and luma.gl 9.4.x: the layer uses
  deck.gl internals (listed in the design notes). Test a new minor before
  widening the range.

## Next

1. a5view could load this package instead of its own copy of the lazy
   renderer, one renderer for both.
2. Propose to deck.gl: abort loading tiles that left the view whatever the
   number in flight (here an override of `Tileset2D.update`).
3. From the design notes, not yet taken: fetching apart from decoding,
   instanced pentagons, a tile-size sweep for 64-byte rows.
