// A region the cells are drawn in: polygons as outer rings, covered at each
// level by a5-js (`polygonToCells`, containment "overlapping", so every cell
// that touches a ring is in and a cell is drawn whole or not at all). Used in
// the workers to drop a tile's rows outside the region, and to tell which
// tiles cannot hold any of its cells.
//
// A covering is held as ranges of cell ids at its level: a stored cell `c`
// (its lowest set bit `t`, the resolution tag) holds the cells of the level
// whose ids lie strictly between c - t and c + t, as a5-js orders its slots
// for resolutions 2 to 29. Coarser stored cells (a whole origin or quintant)
// are replaced by their children at resolution 2 first.

import { cellToChildren, getResolution, isCompactionMarker, polygonToCells, union } from "a5-js";
import type { Box } from "./format.js";

/** What a5-js's polygonToCells takes (its point type, branded degrees, is not exported). */
type A5Polygon = Parameters<typeof polygonToCells>[0];

/** A ring of `[longitude, latitude]` points in degrees, open or closed. */
export type Ring = [number, number][];

const U64_MAX = (1n << 64n) - 1n;

/** The box of a ring (rings are taken not to cross the antimeridian). */
export function ringBox(ring: Ring): Box {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const [x, y] of ring) {
    if (x < w) w = x;
    if (x > e) e = x;
    if (y < s) s = y;
    if (y > n) n = y;
  }
  return [w, s, e, n];
}

/** Rings checked: at least three distinct points each, finite coordinates. */
export function checkRings(rings: Ring[]): Ring[] {
  if (!Array.isArray(rings) || !rings.length) throw new Error("region: give at least one ring");
  return rings.map((r, i) => {
    if (!Array.isArray(r) || r.length < 3) throw new Error(`region: ring ${i} has fewer than three points`);
    for (const p of r) {
      if (!Array.isArray(p) || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) throw new Error(`region: ring ${i} has a point that is not [longitude, latitude]`);
    }
    return r.map((p) => [p[0], p[1]] as [number, number]);
  });
}

/** A covering (compacted cells, with or without its compaction marker) as
 * sorted, merged, inclusive id ranges `[lo0, hi0, lo1, hi1, ...]` of the
 * cells of its level. */
export function coveringRanges(cells: ArrayLike<bigint>): BigUint64Array {
  const runs: [bigint, bigint][] = [];
  const add = (c: bigint) => {
    if (c === 0n) {
      runs.push([0n, U64_MAX]);
      return;
    }
    if (getResolution(c) < 2) {
      for (const k of cellToChildren(c, 2)) add(k);
      return;
    }
    const t = c & -c;
    runs.push([c - t + 1n, c + t - 1n]);
  };
  for (let i = 0; i < cells.length; i++) if (!isCompactionMarker(cells[i])) add(cells[i]);
  runs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const out: bigint[] = [];
  for (const [lo, hi] of runs) {
    const k = out.length;
    if (k && lo <= out[k - 1] + 1n) {
      if (hi > out[k - 1]) out[k - 1] = hi;
    } else out.push(lo, hi);
  }
  return BigUint64Array.from(out);
}

/** The ranges of a level covering a ring. */
export function ringRanges(ring: Ring, lod: number): BigUint64Array {
  return coveringRanges(polygonToCells(ring as unknown as A5Polygon, lod, { containment: "overlapping" }));
}

/** The ranges of several rings' coverings, unioned. */
export function regionRanges(rings: Ring[], lod: number): BigUint64Array {
  let cov: BigUint64Array | null = null;
  for (const r of rings) {
    const c = polygonToCells(r as unknown as A5Polygon, lod, { containment: "overlapping" });
    cov = cov ? union(cov, c) : c;
  }
  return coveringRanges(cov ?? []);
}

/** The first range (its index in pairs) whose upper end is at least `x`. */
function firstRange(ranges: BigUint64Array, x: bigint): number {
  let lo = 0, hi = ranges.length / 2;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ranges[2 * mid + 1] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Whether any id in `[min, max]` lies in the ranges. */
export function rangesMeet(ranges: BigUint64Array, min: bigint, max: bigint): boolean {
  const k = firstRange(ranges, min);
  return k < ranges.length / 2 && ranges[2 * k] <= max;
}

/**
 * Mark the rows whose cells lie in the ranges (`keep[i] = 1`), leaving the
 * others as they are, so several rings' ranges mark one array. One pass for
 * rows in ascending id order, as a tile's are; a row out of order is looked
 * up afresh.
 */
export function markRows(cells: BigUint64Array, ranges: BigUint64Array, keep: Uint8Array): void {
  const m = ranges.length / 2;
  if (!m || !cells.length) return;
  let k = firstRange(ranges, cells[0]);
  let prev = cells[0];
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    if (c < prev) k = firstRange(ranges, c);
    prev = c;
    while (k < m && ranges[2 * k + 1] < c) k++;
    if (k < m && ranges[2 * k] <= c) keep[i] = 1;
  }
}
