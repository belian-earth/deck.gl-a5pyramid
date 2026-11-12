// The browser test's page: draws a fixture set with plain deck.gl and
// exposes check(), which reads back the drawn pixel at cells' centres and
// compares it with the colouring's formula on the CPU.

import { Deck, MapView } from "@deck.gl/core";
import { MaskExtension } from "@deck.gl/extensions";
import { SolidPolygonLayer } from "@deck.gl/layers";
import { cellToLonLat } from "a5-js";
import {
  A5PyramidLayer, PyramidSource, bandColouring, isSettled, pcaBasis, pcaColouring,
  type Colouring, type PcaBasis, type RenderStats,
} from "../../src/index.js";

const decode = (c: number) => Math.sign(c) * (c / 127.5) ** 2;

async function show(name: string, cellPx: number) {
  const source = await PyramidSource.open(`/${name}/`);
  let stats: RenderStats | null = null;
  let basis: PcaBasis | null = null;
  let colouring: Colouring = bandColouring({ ranges: [[-100, 100], [-100, 100], [-40, 80]], version: 0 });
  const deck = new Deck({
    canvas: "map",
    views: new MapView({}),
    initialViewState: { longitude: 117.8, latitude: 4.96, zoom: 15.6 },
    controller: false,
    layers: [],
  });
  const layer = () => new A5PyramidLayer({ id: "p", source, colouring, cellPx, onStats: (s: RenderStats) => (stats = s) });
  deck.setProps({ layers: [layer()] });
  const t0 = performance.now();
  while (!(stats && isSettled(stats, source)) && performance.now() - t0 < 15000) await new Promise((r) => setTimeout(r, 50));
  if (source.pca && stats) {
    basis = pcaBasis((stats as RenderStats).tiles);
    colouring = pcaColouring(basis!);
    deck.setProps({ layers: [layer()] });
    await new Promise((r) => setTimeout(r, 300));
  }
  return { deck, source, stats: stats as RenderStats | null, basis };
}

/** Pixels at picked cells' centres against the formula: the worst channel error. */
async function check(name: string) {
  const { deck, source, stats, basis } = await show(name, 8);
  const gl = (deck as any).device.gl as WebGL2RenderingContext;
  const canvas = (deck as any).canvas as HTMLCanvasElement;
  const vp = deck.getViewports()[0];
  const dpr = canvas.width / canvas.clientWidth;
  let worst = 0, cells = 0;
  for (let y = 40; y < 560; y += 37) {
    for (let x = 40; x < 760; x += 41) {
      const info = deck.pickObject({ x, y, radius: 0 });
      const o = info?.object as { cell: bigint; values: number[] } | undefined;
      if (!o) continue;
      const [cx, cy] = vp.project(cellToLonLat(o.cell) as unknown as [number, number]);
      if (deck.pickObject({ x: cx, y: cy, radius: 0 })?.object?.cell !== o.cell) continue;
      let want: number[];
      if (basis) {
        want = [0, 1, 2].map((ch) => {
          let s = 0;
          for (let k = 0; k < basis.d; k++) s += basis.axes[ch * basis.d + k] * (decode(o.values[k]) - basis.mean[k]);
          return Math.round(Math.min(255, Math.max(0, 127.5 + (127.5 * s) / (2 * Math.sqrt(basis.variance[ch])))));
        });
      } else {
        const r: [number, number][] = [[-100, 100], [-100, 100], [-40, 80]];
        want = r.map(([lo, hi], k) => Math.round(Math.min(1, Math.max(0, (o.values[k] - lo) / (hi - lo))) * 255));
      }
      deck.redraw("check");
      const px = new Uint8Array(4);
      gl.readPixels(Math.round(cx * dpr), canvas.height - Math.round(cy * dpr) - 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      worst = Math.max(worst, ...want.map((w, k) => Math.abs(w - px[k])));
      cells++;
    }
  }
  const out = { name, lod: stats?.lod ?? null, tiles: stats?.tilesLoaded ?? 0, cells, worst };
  deck.finalize();
  source.close();
  return out;
}

/**
 * Drawn with a MaskExtension whose mask is the view's left half: the cells
 * are drawn there and nowhere right of it. Counts of drawn pixels (alpha
 * above 0) sampled on each side, a margin of 20 px left out at the edge.
 */
async function masked(name: string, withMask = true, pickable = true) {
  const source = await PyramidSource.open(`/${name}/`);
  let stats: RenderStats | null = null;
  const deck = new Deck({
    canvas: "map",
    views: new MapView({}),
    initialViewState: { longitude: 117.8, latitude: 4.96, zoom: 15.6 },
    controller: false,
    layers: [],
  });
  await new Promise((r) => setTimeout(r, 50));
  const vp = deck.getViewports()[0];
  const half = [[0, 0], [400, 0], [400, 600], [0, 600], [0, 0]].map(([x, y]) => vp.unproject([x, y]) as [number, number]);
  const colouring = bandColouring({ ranges: [[-100, 100], [-100, 100], [-40, 80]], version: 0 });
  deck.setProps({
    layers: [
      new SolidPolygonLayer({ id: "half", operation: "mask", data: [{ polygon: half }], getPolygon: (d: { polygon: [number, number][] }) => d.polygon }),
      new A5PyramidLayer({ id: "p", source, colouring, cellPx: 8, pickable, ...(withMask ? { extensions: [new MaskExtension()], maskId: "half" } : {}), onStats: (s: RenderStats) => (stats = s) } as any),
    ],
  });
  const t0 = performance.now();
  while (!(stats && isSettled(stats, source)) && performance.now() - t0 < 15000) await new Promise((r) => setTimeout(r, 50));
  deck.redraw("masked");
  await new Promise((r) => setTimeout(r, 300));
  deck.redraw("masked");
  const gl = (deck as any).device.gl as WebGL2RenderingContext;
  const canvas = (deck as any).canvas as HTMLCanvasElement;
  const dpr = canvas.width / canvas.clientWidth;
  let left = 0, right = 0;
  const px = new Uint8Array(4);
  const colours: Record<string, number[]> = {};
  for (let y = 20; y < 580; y += 20)
    for (let x = 10; x < 790; x += 20) {
      if (Math.abs(x - 400) < 20) continue;
      gl.readPixels(Math.round(x * dpr), canvas.height - Math.round(y * dpr) - 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      if (px[3] > 0) x < 400 ? left++ : right++;
      if (x < 380 && px[3] > 0) colours[`${x},${y}`] = [px[0], px[1], px[2], px[3]];
    }
  const out = { name, tiles: (stats as RenderStats | null)?.tilesLoaded ?? 0, left, right, colours };
  deck.finalize();
  source.close();
  return out;
}

(window as any).check = check;
(window as any).masked = masked;

