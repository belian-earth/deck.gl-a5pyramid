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
// The base level, drawn under finer levels still loading, takes a Cover:
// the id runs of the finer levels' loaded tiles (a tile is a run of
// consecutive cell ids, sea between its cells included). A base cell is not
// drawn when the finer cell at its centre falls in a loaded run, since the
// finer level then has its data there or knows there is none (a mask
// texture, a byte a cell). Drawn whole, its 1.4 km cells stood out past the
// coast, over sea the finer cells leave empty; masked by the tiles' boxes
// instead, it left gaps where a box reached over tiles not yet loaded.
// The finer cell at a centre is an a5-js lookup (about 3.5 us), made only
// for cells in view, kept per level, and at most LOOKUPS_PER_FRAME a frame:
// a cell not yet looked up stays drawn.
//
// As deck.gl-raster colours its rasters (a texture of codes, the rendering
// in a shader module), with the work per vertex rather than per pixel: a
// cell is flat, and at one to two pixels a cell there are about as many
// vertices as pixels.

import type { UpdateParameters } from "@deck.gl/core";
import { SolidPolygonLayer } from "@deck.gl/layers";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import { WEIGHT_VEC4S, type Colouring } from "./colour.js";
import { lonLatToCell } from "a5-js";
import type { Box } from "./format.js";
import { boxKey, viewportBox } from "./tiling.js";
import { MAX_INPUTS, type PackedTile } from "./tile-core.js";

const ROW = MAX_INPUTS / 4;
/** Cells a row of a mask texture (one byte each). */
const MASK_WIDTH = 2048;

/** The finer levels' loaded runs; `version` changes with them. */
export interface Cover {
  version: number;
  /** Per level: sorted [lo, hi] pairs of cellKey()s, a loaded tile each. */
  runs: Map<number, Float64Array>;
}

/** A cell id as an ordered number: exact for cells to resolution 25 or so,
 * whose lowest 11 bits are zero. */
export function cellKey(id: bigint): number {
  return Number(id >> 11n);
}

export function inRuns(runs: Float64Array, k: number): boolean {
  // the last pair starting at or before k
  let lo = 0, hi = runs.length / 2 - 1, at = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (runs[2 * mid] <= k) {
      at = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return at >= 0 && k <= runs[2 * at + 1];
}

function inBox(lon: number, lat: number, b: Box): boolean {
  if (lat < b[1] || lat > b[3]) return false;
  return b[0] <= b[2] ? lon >= b[0] && lon <= b[2] : lon >= b[0] || lon <= b[2];
}

const LOOKUPS_PER_FRAME = 1500;
const lookups = { at: -Infinity, left: 0 };
function takeLookup(): boolean {
  const now = performance.now();
  if (now - lookups.at > 12) {
    lookups.at = now;
    lookups.left = LOOKUPS_PER_FRAME;
  }
  return lookups.left-- > 0;
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
  /** For the base level: what covers it now. */
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
  maskView?: string;
  centres?: Float64Array;
  /** Per finer level, the cellKey of the cell at each centre (NaN: not yet). */
  keys?: Map<number, Float64Array>;
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
    const masked = !!cv && cv.runs.size > 0;
    const viewKey = masked ? boxKey(viewportBox(this.context.viewport)) : "";
    if (masked && (state.maskCover !== cv || state.maskVersion !== cv.version || state.maskView !== viewKey)) this.updateMask(cv, viewKey);
    model.shaderInputs.setProps({
      a5colour: {
        weights: c.weights, offset: c.offset, decode: c.decode ? 1 : 0, texels: t.texels, perRow: t.perRow,
        masked: masked ? 1 : 0, a5codes: codes, a5mask: (masked && state.mask) || codes,
      },
    });
    super.draw(opts);
  }

  /** Mask out the cells in view whose centres lie in a loaded finer run. */
  private updateMask(cv: Cover, viewKey: string) {
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
      state.keys = new Map();
    }
    const rows = Math.max(1, Math.ceil(t.n / MASK_WIDTH));
    const bytes = state.maskBytes ?? new Uint8Array(rows * MASK_WIDTH).fill(255);
    state.maskBytes = bytes;
    const c = state.centres;
    const view = viewportBox(this.context.viewport);
    const levels = [...cv.runs].filter(([lod]) => lod > t.lod);
    let complete = true;
    for (let i = 0; i < t.n; i++) {
      const lon = c[2 * i], lat = c[2 * i + 1];
      // out of view: left as it is
      if (!inBox(lon, lat, view)) continue;
      let hidden = false;
      for (const [lod, runs] of levels) {
        let keys = state.keys!.get(lod);
        if (!keys) state.keys!.set(lod, (keys = new Float64Array(t.n).fill(NaN)));
        let k = keys[i];
        if (Number.isNaN(k)) {
          if (!takeLookup()) {
            complete = false;
            continue;
          }
          k = keys[i] = cellKey(lonLatToCell([lon, lat] as unknown as Parameters<typeof lonLatToCell>[0], lod));
        }
        if (inRuns(runs, k)) {
          hidden = true;
          break;
        }
      }
      bytes[i] = hidden ? 0 : 255;
    }
    state.mask ??= this.context.device.createTexture({
      format: "r8unorm",
      width: MASK_WIDTH,
      height: rows,
      mipLevels: 1,
      sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
    });
    state.mask.writeData(bytes);
    state.maskCover = cv;
    state.maskView = viewKey;
    // unfinished (lookups left for the next frames), or the view moved on:
    // made again at the next draw
    state.maskVersion = complete ? cv.version : -1;
    if (!complete) this.setNeedsRedraw();
  }

  finalizeState(context: Parameters<SolidPolygonLayer["finalizeState"]>[0]) {
    super.finalizeState(context);
    const state = this.state as unknown as MaskState & { codes?: Texture };
    state.codes?.destroy();
    state.mask?.destroy();
  }
}
