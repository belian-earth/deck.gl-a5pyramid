import { describe, expect, it } from "vitest";
import { a5Resolution, boxKey, cellPixels, cellSize, metresPerPixel, pickLod, viewportBox } from "../src/tiling.js";

describe("the pixel rule", () => {
  it("has Web Mercator's scale at 512-pixel tiles", () => {
    // MapLibre's Sabah view: zoom 7.7 at 5.3 N, 375 m a CSS pixel
    expect(metresPerPixel(7.7, 5.3)).toBeCloseTo(374.8, 0);
    expect(metresPerPixel(0, 0)).toBeCloseTo(78271.5, 0);
  });

  it("takes the coarsest resolution whose cells fit the target", () => {
    const view = { zoom: 7.7, latitude: 5.3 };
    expect(a5Resolution(view, 1)).toBe(14);
    expect(a5Resolution(view, 2)).toBe(13);
    expect(cellPixels(14, view)).toBeCloseTo(0.95, 2);
    expect(cellPixels(13, view)).toBeCloseTo(1.9, 1);
    for (const px of [0.5, 1, 2, 4, 8]) {
      for (const zoom of [3, 6.4, 9.1, 12.7, 16]) {
        const v = { zoom, latitude: 40 };
        const r = a5Resolution(v, px);
        expect(cellPixels(r, v)).toBeLessThanOrEqual(px);
        if (r > 0) expect(cellPixels(r - 1, v)).toBeGreaterThan(px);
      }
    }
  });

  it("sizes cells from their (equal) areas", () => {
    // each resolution a quarter of the area of the one before
    for (let r = 2; r < 30; r++) expect(cellSize(r) / cellSize(r + 1)).toBeCloseTo(2, 6);
  });

  it("picks the finest level at or below a resolution, else the coarsest", () => {
    const lods = [2, 3, 4, 10, 11];
    expect(pickLod(11, lods)).toBe(11);
    expect(pickLod(9, lods)).toBe(4);
    expect(pickLod(1, lods)).toBe(2);
  });
});

describe("view boxes", () => {
  it("pad the view, wrap the antimeridian and take the globe without bounds", () => {
    expect(viewportBox({ zoom: 1, getBounds: () => [10, 0, 20, 10] })).toEqual([9, -1, 21, 11]);
    const b = viewportBox({ zoom: 1, getBounds: () => [170, 0, 190, 10] });
    expect(b[0]).toBeGreaterThan(b[2]);
    expect(viewportBox({ zoom: 1 })).toEqual([-180, -90, 180, 90]);
  });

  it("keep a key for nearby views", () => {
    expect(boxKey([10, 0, 20, 10])).toBe(boxKey([10.01, 0.01, 20.01, 10.01]));
    expect(boxKey([10, 0, 20, 10])).not.toBe(boxKey([12, 0, 22, 10]));
  });
});
