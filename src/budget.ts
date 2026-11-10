// What a view may read and draw, and the level that keeps within it.

/** The most a view may read and draw: compressed bytes and cells. */
export interface Budget {
  bytes: number;
  cells: number;
}

/** A view's budget by default: about 3 s from a warm CDN edge at 10 to
 * 15 MB/s, and some 200 MB of geometry in memory. */
export const DEFAULT_BUDGET: Budget = { bytes: 40e6, cells: 1.5e6 };

/** What the base level may hold, over the whole set. */
export const BASE_BUDGET: Budget = { bytes: 8e6, cells: 150e3 };

/** A level on screen keeps this much over the budget before it gives way,
 * so a view near the limit does not flip between two levels as it pans. */
export const BUDGET_SLACK = 1.25;

/** What reading a level for a view costs: its tiles' rows and compressed bytes. */
export interface Cost {
  rows: number;
  bytes: number;
}

export interface LevelChoice {
  lod: number;
  /** Why a level coarser than the target was chosen. */
  capped: "bytes" | "cells" | null;
  /** The chosen level's cost, or null while it is unknown. */
  cost: Cost | null;
}

/**
 * The level to draw: `target`, or the finest coarser one of `lods`
 * (ascending, contiguous) within `budget`. `cost` gives a level's cost for
 * the view, or null while it is unknown (its shards still opening): such a
 * level is taken as it is, to be judged once known. A level `onScreen`
 * keeps BUDGET_SLACK over the budget.
 */
export function chooseLevel(
  target: number,
  lods: number[],
  cost: (lod: number) => Cost | null,
  budget: Budget,
  onScreen: (lod: number) => boolean = () => false,
): LevelChoice {
  let lod = target;
  let capped: LevelChoice["capped"] = null;
  for (;;) {
    const c = cost(lod);
    if (!c) return { lod, capped, cost: null };
    const slack = onScreen(lod) ? BUDGET_SLACK : 1;
    const over = c.bytes > budget.bytes * slack ? "bytes" : c.rows > budget.cells * slack ? "cells" : null;
    if (!over || lod <= lods[0]) return { lod, capped, cost: c };
    capped = over;
    lod = lods[Math.max(0, lods.indexOf(lod) - 1)];
  }
}
