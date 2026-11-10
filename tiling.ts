// The A5 resolution for a view, and viewport boxes (the boxes ported from
// a5view's tiling.js, inst/htmlwidgets/lib/a5view-tiling/tiling.js,
// 2026-11-09).
//
// a5view chose the resolution from a table of (zoom, resolution) anchors,
// which gives cells of 0.6 to 2.3 CSS pixels depending on the zoom. Here it
// comes from the cells' size on screen: A5 cells are equal-area, so a
// resolution's cell size is exact (the square root of its area), and the
// view's scale is Web Mercator's at its centre. The coarsest resolution
// whose cells are at most `px` pixels across is drawn, as deck.gl-raster
// picks the coarsest overview at least as fine as the screen.

import { cellArea } from "a5-js";
import type { Box } from "./format";

/** Fraction of the view added on each side when choosing tiles. */
export const VIEWPORT_BUFFER = 0.1;

interface ViewportLike {
  zoom: number;
  latitude?: number;
  getBounds?: () => number[];
}

/** Web Mercator's equator, metres (WGS 84 semi-major axis). */
const EQUATOR_M = 2 * Math.PI * 6378137;

/** A cell's size at each resolution, metres: the square root of its area. */
const CELL_M = Array.from({ length: 31 }, (_, r) => Math.sqrt(cellArea(r)));

export function cellSize(resolution: number): number {
  return CELL_M[Math.max(0, Math.min(30, resolution))];
}

/** Metres a CSS pixel spans at a zoom and latitude (512-pixel tiles, as
 * MapLibre's and deck.gl's zooms count them). */
export function metresPerPixel(zoom: number, latitude = 0): number {
  const cos = Math.max(0.01, Math.cos((latitude * Math.PI) / 180));
  return (EQUATOR_M * cos) / (512 * 2 ** zoom);
}

/** CSS pixels across a cell of `resolution` at the view's centre. */
export function cellPixels(resolution: number, viewport: ViewportLike): number {
  return cellSize(resolution) / metresPerPixel(viewport.zoom || 0, viewport.latitude || 0);
}

/** The coarsest A5 resolution whose cells are at most `px` CSS pixels
 * across at the view's centre. */
export function a5Resolution(viewport: ViewportLike, px = 1): number {
  const limit = px * metresPerPixel(viewport.zoom || 0, viewport.latitude || 0);
  for (let r = 0; r <= 30; r++) if (CELL_M[r] <= limit) return r;
  return 30;
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
