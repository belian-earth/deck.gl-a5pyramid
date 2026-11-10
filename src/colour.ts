// How tiles are coloured. The GPU computes each cell's colour from the codes
// the tile uploaded once (codes-layer.ts): red, green and blue are each an
// affine function of the cell's codes, or of the values they decode to,
// clamped to 0..1. A colouring is that function's weights and offsets, about
// 1 KB of uniforms, so a new colouring redraws without touching the cells.

import { MAX_INPUTS } from "./tile-core.js";

export interface Stretch {
  /** `[low, high]` code per band, in the order of the set's columns. */
  ranges: [number, number][];
  /** Changes whenever the ranges do. */
  version: number;
}

/** Vec4s of weights: one row of MAX_INPUTS / 4 a channel. */
export const WEIGHT_VEC4S = 3 * (MAX_INPUTS / 4);

export interface Colouring {
  /** Changes whenever the colours would. */
  version: number;
  /**
   * Weight of input k in channel ch (0 red, 1 green, 2 blue) at
   * weightIndex(ch, k): the shader's WEIGHT_VEC4S vec4s, a channel's row
   * of them dotted with the cell's codes four at a time.
   */
  weights: Float32Array;
  /** Red, green and blue at all-zero inputs, and the alpha, 0..1. */
  offset: [number, number, number, number];
  /** Inputs are the decoded AEF values, sign(x) (x / 127.5)^2, not the codes. */
  decode: boolean;
}

export function weightIndex(ch: number, k: number): number {
  return (ch * (MAX_INPUTS / 4) + (k >> 2)) * 4 + (k & 3);
}

let nextVersion = 1;

/** A colouring from its parts; `weights` starts at zero. */
export function colouring(offset: [number, number, number, number], decode: boolean, fill?: (w: Float32Array) => void): Colouring {
  const weights = new Float32Array(WEIGHT_VEC4S * 4);
  fill?.(weights);
  return { version: nextVersion++, weights, offset, decode };
}

/** The first three bands as red, green and blue, each stretched linearly
 * between two codes. */
export function bandColouring(stretch: Stretch): Colouring {
  const offset: [number, number, number, number] = [0, 0, 0, 1];
  return colouring(offset, false, (w) => {
    for (let ch = 0; ch < 3 && ch < stretch.ranges.length; ch++) {
      const [lo, hi] = stretch.ranges[ch];
      const span = hi - lo || 1;
      w[weightIndex(ch, ch)] = 1 / span;
      offset[ch] = -lo / span;
    }
  });
}

/** One grey for every cell: a set coloured from its view (PCA) until the
 * view's first tiles have loaded. */
export function neutralColouring(): Colouring {
  return colouring([150 / 255, 150 / 255, 150 / 255, 200 / 255], false);
}

/** An AEF code as the embedding value it stands for. */
export function decodeAef(code: number): number {
  const x = code / 127.5;
  return Math.sign(code) * x * x;
}
