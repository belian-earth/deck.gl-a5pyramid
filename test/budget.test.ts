import { describe, expect, it } from "vitest";
import { BUDGET_SLACK, chooseLevel, type Cost } from "../src/budget.js";

const lods = Array.from({ length: 17 }, (_, i) => i + 2);
// a full view: four times the cells (and bytes) a level finer
const full = (lod: number): Cost => ({ rows: 4 ** (lod - 10) * 1e4, bytes: 4 ** (lod - 10) * 1e6 });
const budget = { bytes: 40e6, cells: 1.5e6 };

describe("chooseLevel", () => {
  it("keeps the target within the budget", () => {
    expect(chooseLevel(12, lods, full, budget)).toEqual({ lod: 12, capped: null, cost: full(12) });
  });

  it("steps coarser past it, and says why", () => {
    // 13: 64 MB, 640 k cells; 12: 16 MB
    expect(chooseLevel(13, lods, full, budget)).toMatchObject({ lod: 12, capped: "bytes" });
    const many = (lod: number): Cost => ({ rows: 4 ** (lod - 10) * 1e6, bytes: 1 });
    expect(chooseLevel(12, lods, many, budget)).toMatchObject({ lod: 10, capped: "cells" });
  });

  it("takes a level of unknown cost as it is", () => {
    const unknown = (lod: number) => (lod >= 14 ? null : full(lod));
    expect(chooseLevel(15, lods, unknown, budget)).toEqual({ lod: 15, capped: null, cost: null });
  });

  it("lets a level on screen keep its slack", () => {
    const near = (lod: number): Cost => ({ rows: 1, bytes: lod === 13 ? budget.bytes * (1 + BUDGET_SLACK) / 2 : 1 });
    expect(chooseLevel(13, lods, near, budget).lod).toBe(12);
    expect(chooseLevel(13, lods, near, budget, (l) => l === 13).lod).toBe(13);
  });

  it("stops at the coarsest level", () => {
    const huge = (): Cost => ({ rows: 1e12, bytes: 1e12 });
    expect(chooseLevel(5, lods, huge, budget)).toMatchObject({ lod: 2, capped: "bytes" });
  });
});
