// Zoom to A5 resolution and viewport boxes, ported from a5view's tiling.js
// (inst/htmlwidgets/lib/a5view-tiling/tiling.js, 2026-11-09).

import type { Box } from "./format";

/** (zoom, resolution) anchors, interpolated between and extrapolated beyond. */
export const ZOOM_TO_RES: [number, number][] = [
  [0, 7],
  [3, 10],
  [7, 14],
  [10, 15],
  [16, 22],
];

/** Fraction of the view added on each side when choosing tiles. */
export const VIEWPORT_BUFFER = 0.1;

interface ViewportLike {
  zoom: number;
  latitude?: number;
  getBounds?: () => number[];
}

/**
 * The A5 resolution for a view: the anchors' interpolation at the zoom, with
 * a small boost at high latitudes (Mercator pixels cover less ground there)
 * and `bias` resolutions added (negative for coarser cells).
 */
export function a5Resolution(viewport: ViewportLike, bias = 0): number {
  let z = viewport.zoom || 0;
  const lat = viewport.latitude || 0;
  z += Math.log(1 / Math.max(0.05, Math.cos((lat * Math.PI) / 180)));
  const t = ZOOM_TO_RES;
  let r: number;
  if (z <= t[0][0]) {
    const slope = (t[1][1] - t[0][1]) / (t[1][0] - t[0][0]);
    r = t[0][1] + slope * (z - t[0][0]);
  } else {
    let i = 1;
    while (i < t.length && z > t[i][0]) i++;
    if (i === t.length) {
      const n = t.length;
      const slope = (t[n - 1][1] - t[n - 2][1]) / (t[n - 1][0] - t[n - 2][0]);
      r = t[n - 1][1] + slope * (z - t[n - 1][0]);
    } else {
      const [z0, r0] = t[i - 1];
      const [z1, r1] = t[i];
      r = r0 + ((z - z0) / (z1 - z0)) * (r1 - r0);
    }
  }
  return Math.max(0, Math.floor(r + bias));
}

/** The finest LOD of `lods` (ascending) at or below `r`, else the coarsest. */
export function pickLod(r: number, lods: number[]): number {
  let picked = lods[0];
  for (const l of lods) {
    if (l <= r) picked = l;
    else break;
  }
  return picked;
}

/**
 * The view's box, widened by VIEWPORT_BUFFER on each side; the whole globe
 * when the bounds are missing (a globe view) or span every longitude. May
 * cross the antimeridian (`west > east`).
 */
export function viewportBox(viewport: ViewportLike): Box {
  let b: number[] | null = null;
  try {
    b = viewport.getBounds ? viewport.getBounds() : null;
  } catch {
    b = null;
  }
  if (!b || b.length !== 4 || !b.every(Number.isFinite)) return [-180, -90, 180, 90];
  let [west, south, east, north] = b;
  south = Math.max(-90, south);
  north = Math.min(90, north);
  const lonSpan = east - west;
  if (lonSpan >= 360) return [-180, south, 180, north];
  const lonBuf = lonSpan * VIEWPORT_BUFFER;
  const latBuf = (north - south) * VIEWPORT_BUFFER;
  south = Math.max(-90, south - latBuf);
  north = Math.min(90, north + latBuf);
  if (lonSpan + 2 * lonBuf >= 360) return [-180, south, 180, north];
  west -= lonBuf;
  east += lonBuf;
  const wrap = (x: number) => ((((x + 180) % 360) + 360) % 360) - 180;
  return [wrap(west), south, wrap(east), north];
}

/** A key that stays the same for nearby views, so tile selection is memoised. */
export function boxKey(box: Box): string {
  const lonSpan = box[2] >= box[0] ? box[2] - box[0] : 360 - (box[0] - box[2]);
  const span = Math.max(lonSpan, box[3] - box[1], 1e-3);
  const q = span / 32;
  return box.map((x) => Math.round(x / q)).join("/") + "@" + q.toFixed(6);
}
