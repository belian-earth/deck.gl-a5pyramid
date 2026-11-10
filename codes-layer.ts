// A tile's cells as polygons coloured on the GPU from their codes.
//
// A SolidPolygonLayer whose fill colour is computed in the vertex shader:
// the tile's codes go to the GPU once, as an RGBA8 texture (four codes a
// texel, a cell's texels in one row, tile-core's codesLayout), and the
// colouring (colour.ts) arrives as uniforms. The vertex shader finds its
// cell from deck.gl's per-vertex row index (the attribute picking already
// uses), reads the cell's codes and dots them, four at a time, with each
// channel's weights. A new colouring changes the uniforms and redraws; the
// geometry and the codes stay on the GPU as they are.
//
// A coarser level drawn under a finer one still loading (the backdrop, the
// base) can take a Cover: the boxes of the finer level's loaded tiles. Its
// cells whose centres lie in one are not drawn (a mask texture, a byte a
// cell, made again when the cover changes), so it fills only where nothing
// finer has arrived; drawn whole, its large cells stood out past the coast,
// over sea the finer cells leave empty.
//
// As deck.gl-raster colours its rasters (a texture of codes, the rendering
// in a shader module), with the work per vertex rather than per pixel: a
// cell is flat, and at one to two pixels a cell there are about as many
// vertices as pixels.

import type { UpdateParameters } from "@deck.gl/core";
import { SolidPolygonLayer } from "@deck.gl/layers";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import { WEIGHT_VEC4S, type Colouring } from "./colour";
import type { Box } from "./format";
import { MAX_INPUTS, type PackedTile } from "./tile-core";

const ROW = MAX_INPUTS / 4;
/** Cells a row of a mask texture (one byte each). */
const MASK_WIDTH = 2048;

/** Boxes of the finer level's loaded tiles; `version` changes with them. */
export interface Cover {
  version: number;
  boxes: Box[];
}

function inBox(lon: number, lat: number, b: Box): boolean {
  if (lat < b[1] || lat > b[3]) return false;
  return b[0] <= b[2] ? lon >= b[0] && lon <= b[2] : lon >= b[0] || lon <= b[2];
}

/** A cover's boxes bucketed on a GRID x GRID grid over their extent, so a
 * point is tested against the few boxes near it; built once per version. */
const GRID = 32;
interface CoverGrid { version: number; w: number; s: number; dx: number; dy: number; cells: Box[][]; wide: Box[] }
const grids = new WeakMap<Cover, CoverGrid>();
function coverGrid(cv: Cover): CoverGrid {
  const known = grids.get(cv);
  if (known && known.version === cv.version) return known;
  // boxes across the antimeridian are tested for every point
  const wide = cv.boxes.filter((b) => b[0] > b[2]);
  const flat = cv.boxes.filter((b) => b[0] <= b[2]);
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const b of flat) {
    w = Math.min(w, b[0]); s = Math.min(s, b[1]); e = Math.max(e, b[2]); n = Math.max(n, b[3]);
  }
  const dx = (e - w) / GRID || 1, dy = (n - s) / GRID || 1;
  const cells: Box[][] = Array.from({ length: GRID * GRID }, () => []);
  const clamp = (k: number) => Math.max(0, Math.min(GRID - 1, k));
  for (const b of flat) {
    for (let gy = clamp(Math.floor((b[1] - s) / dy)); gy <= clamp(Math.floor((b[3] - s) / dy)); gy++) {
      for (let gx = clamp(Math.floor((b[0] - w) / dx)); gx <= clamp(Math.floor((b[2] - w) / dx)); gx++) cells[gy * GRID + gx].push(b);
    }
  }
  const g = { version: cv.version, w, s, dx, dy, cells, wide };
  grids.set(cv, g);
  return g;
}

function covered(g: CoverGrid, lon: number, lat: number): boolean {
  for (const b of g.wide) if (inBox(lon, lat, b)) return true;
  const gx = Math.floor((lon - g.w) / g.dx), gy = Math.floor((lat - g.s) / g.dy);
  if (gx < 0 || gy < 0 || gx >= GRID || gy >= GRID) return false;
  for (const b of g.cells[gy * GRID + gx]) if (inBox(lon, lat, b)) return true;
  return false;
}

const glsl = /* glsl */ `\
layout(std140) uniform a5colourUniforms {
  vec4 weights[${WEIGHT_VEC4S}];
  vec4 offset;
  float decode;
  int texels;
  int perRow;
  float masked;
} a5colour;

uniform highp sampler2D a5codes;
uniform highp sampler2D a5mask;

// Whether the cell at row index \`row\` is masked out (its mask byte 0).
bool a5colour_hidden(float row) {
  if (a5colour.masked < 0.5) return false;
  int i = int(row + 0.5);
  return texelFetch(a5mask, ivec2(i % ${MASK_WIDTH}, i / ${MASK_WIDTH}), 0).r < 0.5;
}

// The colour of the cell at row index \`row\` of the tile.
vec3 a5colour_rgb(float row) {
  int i = int(row + 0.5);
  int x0 = (i % a5colour.perRow) * a5colour.texels;
  int y = i / a5colour.perRow;
  vec3 c = a5colour.offset.rgb;
  for (int j = 0; j < ${ROW}; j++) {
    if (j >= a5colour.texels) break;
    // unorm bytes back to the signed codes
    vec4 code = floor(texelFetch(a5codes, ivec2(x0 + j, y), 0) * 255.0 + 0.5);
    code -= step(127.5, code) * 256.0;
    vec4 x = code / 127.5;
    vec4 v = a5colour.decode > 0.5 ? sign(x) * x * x : code;
    c += vec3(dot(v, a5colour.weights[j]), dot(v, a5colour.weights[${ROW} + j]), dot(v, a5colour.weights[${2 * ROW} + j]));
  }
  return clamp(c, 0.0, 1.0);
}
`;

type A5ColourProps = {
  weights: Float32Array;
  offset: [number, number, number, number];
  decode: number;
  texels: number;
  perRow: number;
  masked: number;
  a5codes: Texture;
  a5mask: Texture;
};

const a5colour = {
  name: "a5colour",
  vs: glsl,
  uniformTypes: {
    weights: ["vec4<f32>", WEIGHT_VEC4S],
    offset: "vec4<f32>",
    decode: "f32",
    texels: "i32",
    perRow: "i32",
    masked: "f32",
  },
  // (luma's prop types take no typed array for an array uniform; it writes one)
} as unknown as ShaderModule<A5ColourProps>;

/** The colouring every tile draws with, changed in place by the parent layer. */
export interface Palette {
  colouring: Colouring;
}

interface CodesLayerProps {
  /** For a coarser level under a finer one: what covers it now, if anything. */
  cover?: () => Cover | null;
  /** (Not `tile`: TileLayer gives each sublayer its own `tile`, the tile's header.) */
  packed: PackedTile;
  palette: Palette;
}

interface MaskState {
  mask?: Texture;
  maskBytes?: Uint8Array;
  maskCover?: Cover;
  maskVersion?: number;
  centres?: Float64Array;
}

export class CodesPolygonLayer extends SolidPolygonLayer<any, CodesLayerProps> {
  static layerName = "A5CodesPolygonLayer";

  getShaders(type: "top" | "side") {
    const s = super.getShaders(type);
    return {
      ...s,
      modules: [...s.modules, a5colour],
      inject: {
        ...(s.inject ?? {}),
        // after deck's colour (and picking's hook, which only records the
        // picking colour): the cell's colour, at the colouring's alpha
        // a masked cell goes outside the clip volume, so its triangles are culled
        "vs:#main-end": `vColor = vec4(a5colour_rgb(rowIndexes), a5colour.offset.a * layer.opacity);
  if (a5colour_hidden(rowIndexes)) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);`,
      },
    };
  }

  updateState(params: UpdateParameters<this>) {
    super.updateState(params);
    const { props, oldProps } = params as unknown as { props: CodesLayerProps; oldProps: Partial<CodesLayerProps> };
    if (props.packed !== oldProps.packed) {
      const state = this.state as unknown as { codes?: Texture };
      state.codes?.destroy();
      const t = props.packed;
      const codes = this.context.device.createTexture({
        format: "rgba8unorm",
        width: t.codesWidth,
        height: t.codesRows,
        mipLevels: 1,
        sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
      });
      codes.writeData(t.codes);
      state.codes = codes;
    }
  }

  draw(opts: Parameters<SolidPolygonLayer["draw"]>[0]) {
    const { packed: t, palette, cover } = this.props;
    const c = palette.colouring;
    const state = this.state as unknown as MaskState & { codes?: Texture };
    const codes = state.codes;
    const model = (this.state as { topModel?: { shaderInputs: { setProps(p: object): void } } }).topModel;
    if (!codes || !model) return;
    const cv = cover?.() ?? null;
    const masked = !!cv && cv.boxes.length > 0;
    if (masked && (state.maskCover !== cv || state.maskVersion !== cv.version)) this.updateMask(cv);
    model.shaderInputs.setProps({
      a5colour: {
        weights: c.weights, offset: c.offset, decode: c.decode ? 1 : 0, texels: t.texels, perRow: t.perRow,
        masked: masked ? 1 : 0, a5codes: codes, a5mask: (masked && state.mask) || codes,
      },
    });
    super.draw(opts);
  }

  /** Mask out the cells whose centres lie in a box of the cover. */
  private updateMask(cv: Cover) {
    const t = this.props.packed;
    const state = this.state as unknown as MaskState;
    if (!state.centres) {
      // a cell's centre: the mean of its ring's vertices
      const c = new Float64Array(2 * t.n);
      for (let i = 0; i < t.n; i++) {
        let x = 0, y = 0;
        const a = t.starts[i], b = t.starts[i + 1];
        for (let v = a; v < b; v++) {
          x += t.positions[2 * v];
          y += t.positions[2 * v + 1];
        }
        c[2 * i] = x / (b - a);
        c[2 * i + 1] = y / (b - a);
      }
      state.centres = c;
    }
    const rows = Math.max(1, Math.ceil(t.n / MASK_WIDTH));
    const bytes = state.maskBytes ?? new Uint8Array(rows * MASK_WIDTH);
    state.maskBytes = bytes;
    const c = state.centres;
    const g = coverGrid(cv);
    for (let i = 0; i < t.n; i++) bytes[i] = covered(g, c[2 * i], c[2 * i + 1]) ? 0 : 255;
    state.mask ??= this.context.device.createTexture({
      format: "r8unorm",
      width: MASK_WIDTH,
      height: rows,
      mipLevels: 1,
      sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
    });
    state.mask.writeData(bytes);
    state.maskCover = cv;
    state.maskVersion = cv.version;
  }

  finalizeState(context: Parameters<SolidPolygonLayer["finalizeState"]>[0]) {
    super.finalizeState(context);
    const state = this.state as unknown as MaskState & { codes?: Texture };
    state.codes?.destroy();
    state.mask?.destroy();
  }
}
