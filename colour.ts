// Colouring a tile from its stored codes: each band stretched linearly
// between two codes to 0..255, through a 256-entry table per band, written
// per vertex (deck.gl's binary polygons take one colour per vertex).

import type { PackedTile } from "./tile-core";

export interface Stretch {
  /** `[low, high]` code per band, in the order of the set's columns. */
  ranges: [number, number][];
  /** Changes whenever the ranges do, so tiles are recoloured. */
  version: number;
}

function table([lo, hi]: [number, number]): Uint8Array {
  const out = new Uint8Array(256);
  const span = hi - lo || 1;
  for (let c = -128; c < 128; c++) {
    out[c + 128] = Math.round(Math.min(1, Math.max(0, (c - lo) / span)) * 255);
  }
  return out;
}

/**
 * How tiles are coloured: per-vertex RGBA for a tile, and a version that
 * changes whenever the colours would, so the layer recolours tiles it
 * already holds without reading them again.
 */
export interface Colouring {
  version: number;
  colour(t: PackedTile): Uint8Array;
}

let nextVersion = 1;

/** The first three bands as red, green and blue, each through its stretch. */
export function bandColouring(stretch: Stretch): Colouring {
  return { version: nextVersion++, colour: (t) => colours(t, stretch) };
}

/** One grey for every cell: a set coloured from its view (PCA) until the
 * view's first tiles have loaded. */
export function neutralColouring(): Colouring {
  return {
    version: nextVersion++,
    colour: (t) => {
      const out = new Uint8Array(t.starts[t.n] * 4);
      for (let v = 0; v < out.length; v += 4) {
        out[v] = out[v + 1] = out[v + 2] = 150;
        out[v + 3] = 200;
      }
      return out;
    },
  };
}

const tables = new WeakMap<Stretch, Uint8Array[]>();

/** Per-vertex RGBA for the first three value columns of `t` (grey for a
 * tile without three). */
export function colours(t: PackedTile, stretch: Stretch): Uint8Array {
  if (t.values.length < 3) return neutralColouring().colour(t);
  let luts = tables.get(stretch);
  if (!luts) {
    luts = stretch.ranges.map(table);
    tables.set(stretch, luts);
  }
  const [r, g, b] = [0, 1, 2].map((k) => t.values[k]);
  const [lr, lg, lb] = luts;
  const nv = t.starts[t.n];
  const out = new Uint8Array(nv * 4);
  for (let i = 0; i < t.n; i++) {
    const cr = lr[r[i] + 128], cg = lg[g[i] + 128], cb = lb[b[i] + 128];
    for (let v = t.starts[i]; v < t.starts[i + 1]; v++) {
      out[4 * v] = cr;
      out[4 * v + 1] = cg;
      out[4 * v + 2] = cb;
      out[4 * v + 3] = 255;
    }
  }
  return out;
}

/** An AEF code as the embedding value it stands for. */
export function decodeAef(code: number): number {
  const x = code / 127.5;
  return Math.sign(code) * x * x;
}
