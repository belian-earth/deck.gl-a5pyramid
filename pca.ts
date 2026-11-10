// Principal components of the cells in view, and colours from them.
//
// Each tile arrives with sums of its decoded values (TileStats, from the
// worker), so the covariance of everything in view is a sum over tiles: no
// pass over the cells. The first three principal axes become red, green and
// blue, each stretched over the view's mean +/- 2 standard deviations. A new
// basis is matched to the previous one (axis order and sign), so colours
// shift with the view rather than jump.

import { colouring, weightIndex, type Colouring } from "./colour";
import { AEF_DECODE, MAX_INPUTS, type PackedTile, type TileStats } from "./tile-core";

export interface PcaBasis {
  d: number;
  /** Weighted mean of the decoded values in view. */
  mean: Float64Array;
  /** Red, green and blue axes, row by row (3 x d), unit length. */
  axes: Float64Array;
  /** Variance along each of the three axes, and over all d dimensions. */
  variance: [number, number, number];
  total: number;
  /** The principal component (1-based) each channel shows. */
  rank: [number, number, number];
  /** Tiles and total weight (leaves) the basis comes from. */
  tiles: number;
  weight: number;
  /** Time to combine the sums and solve, ms. */
  ms: number;
}

/** The covariance of the tiles' values, weighted, with their mean. */
export function covariance(stats: TileStats[]): { d: number; w: number; mean: Float64Array; cov: Float64Array } | null {
  if (!stats.length) return null;
  const d = stats[0].d;
  let w = 0;
  const sum = new Float64Array(d);
  const cross = new Float64Array((d * (d + 1)) / 2);
  for (const s of stats) {
    if (s.d !== d) continue;
    w += s.w;
    for (let j = 0; j < d; j++) sum[j] += s.sum[j];
    for (let j = 0; j < cross.length; j++) cross[j] += s.cross[j];
  }
  if (w <= 0) return null;
  const mean = sum.map((x) => x / w);
  const cov = new Float64Array(d * d);
  let at = 0;
  for (let j = 0; j < d; j++) {
    for (let k = j; k < d; k++) {
      const c = cross[at++] / w - mean[j] * mean[k];
      cov[j * d + k] = c;
      cov[k * d + j] = c;
    }
  }
  return { d, w, mean, cov };
}

/**
 * Eigenvalues and eigenvectors of a symmetric matrix (cyclic Jacobi), in
 * descending order of eigenvalue: `vectors` holds vector k at k * n.
 */
export function eigenSymmetric(m: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const a = m.slice();
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  let scale = 0;
  for (let i = 0; i < n * n; i++) scale += a[i] * a[i];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    if (off <= 1e-24 * scale) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  order.forEach((col, k) => {
    values[k] = a[col * n + col];
    for (let i = 0; i < n; i++) vectors[k * n + i] = v[i * n + col];
  });
  return { values, vectors };
}

/**
 * The basis for the tiles in view. With `prev`, the three axes are assigned
 * to the channels they most resemble there, with matching signs, so a small
 * change of view changes the colours a little.
 */
export function pcaBasis(tiles: PackedTile[], prev: PcaBasis | null = null): PcaBasis | null {
  const t0 = performance.now();
  const stats = tiles.map((t) => t.stats).filter((s): s is TileStats => !!s);
  const c = covariance(stats);
  if (!c) return null;
  const { d, w, mean, cov } = c;
  const { values, vectors } = eigenSymmetric(cov, d);
  let total = 0;
  for (let i = 0; i < d; i++) total += Math.max(0, values[i]);

  // channel for each of the first three components, and its sign
  const channel = [0, 1, 2];
  const sign = [1, 1, 1];
  if (prev && prev.d === d) {
    const dots = [0, 1, 2].map((i) => [0, 1, 2].map((j) => {
      let s = 0;
      for (let k = 0; k < d; k++) s += vectors[i * d + k] * prev.axes[j * d + k];
      return s;
    }));
    const usedI = new Set<number>(), usedJ = new Set<number>();
    for (let round = 0; round < 3; round++) {
      let best = -1, bi = 0, bj = 0;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        if (usedI.has(i) || usedJ.has(j)) continue;
        if (Math.abs(dots[i][j]) > best) { best = Math.abs(dots[i][j]); bi = i; bj = j; }
      }
      usedI.add(bi); usedJ.add(bj);
      channel[bi] = bj;
      sign[bi] = dots[bi][bj] < 0 ? -1 : 1;
    }
  } else {
    // first basis: point each axis towards its largest loading, for a
    // repeatable sign
    for (let i = 0; i < 3; i++) {
      let big = 0;
      for (let k = 0; k < d; k++) if (Math.abs(vectors[i * d + k]) > Math.abs(big)) big = vectors[i * d + k];
      sign[i] = big < 0 ? -1 : 1;
    }
  }
  const axes = new Float64Array(3 * d);
  const variance: [number, number, number] = [0, 0, 0];
  const rank: [number, number, number] = [1, 2, 3];
  for (let i = 0; i < 3; i++) {
    const ch = channel[i];
    for (let k = 0; k < d; k++) axes[ch * d + k] = sign[i] * vectors[i * d + k];
    variance[ch] = Math.max(0, values[i]);
    rank[ch] = i + 1;
  }
  return { d, mean, axes, variance, total, rank, tiles: stats.length, weight: w, ms: performance.now() - t0 };
}

/**
 * Colours from a basis, for the GPU: each channel's score stretched over the
 * view's mean +/- 2 standard deviations, as weights on the decoded values.
 */
export function pcaColouring(basis: PcaBasis): Colouring & { basis: PcaBasis } {
  const { d, axes, mean, variance } = basis;
  const offset: [number, number, number, number] = [0, 0, 0, 1];
  const c = colouring(offset, true, (w) => {
    for (let ch = 0; ch < 3; ch++) {
      const sd = Math.sqrt(variance[ch]) || 1;
      // 0..1 across mean +/- 2 sd
      const gain = 1 / (4 * sd);
      let centre = 0;
      for (let k = 0; k < d; k++) centre += axes[ch * d + k] * mean[k];
      offset[ch] = 0.5 - gain * centre;
      for (let k = 0; k < d && k < MAX_INPUTS; k++) w[weightIndex(ch, k)] = axes[ch * d + k] * gain;
    }
  });
  return { ...c, basis };
}

/** A cell's three channel scores, in standard deviations from the view's mean. */
export function scores(basis: PcaBasis, codes: number[]): [number, number, number] {
  const { d, axes, mean, variance } = basis;
  const out: [number, number, number] = [0, 0, 0];
  for (let ch = 0; ch < 3; ch++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += axes[ch * d + k] * (AEF_DECODE[codes[k] + 128] - mean[k]);
    out[ch] = s / (Math.sqrt(variance[ch]) || 1);
  }
  return out;
}
