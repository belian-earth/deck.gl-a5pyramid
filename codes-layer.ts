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
// As deck.gl-raster colours its rasters (a texture of codes, the rendering
// in a shader module), with the work per vertex rather than per pixel: a
// cell is flat, and at one to two pixels a cell there are about as many
// vertices as pixels.

import type { UpdateParameters } from "@deck.gl/core";
import { SolidPolygonLayer } from "@deck.gl/layers";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import { WEIGHT_VEC4S, type Colouring } from "./colour";
import { MAX_INPUTS, type PackedTile } from "./tile-core";

const ROW = MAX_INPUTS / 4;

const glsl = /* glsl */ `\
layout(std140) uniform a5colourUniforms {
  vec4 weights[${WEIGHT_VEC4S}];
  vec4 offset;
  float decode;
  int texels;
  int perRow;
} a5colour;

uniform highp sampler2D a5codes;

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
  a5codes: Texture;
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
  },
  // (luma's prop types take no typed array for an array uniform; it writes one)
} as unknown as ShaderModule<A5ColourProps>;

/** The colouring every tile draws with, changed in place by the parent layer. */
export interface Palette {
  colouring: Colouring;
}

interface CodesLayerProps {
  /** (Not `tile`: TileLayer gives each sublayer its own `tile`, the tile's header.) */
  packed: PackedTile;
  palette: Palette;
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
        "vs:#main-end": "vColor = vec4(a5colour_rgb(rowIndexes), a5colour.offset.a * layer.opacity);",
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
    const { packed: t, palette } = this.props;
    const c = palette.colouring;
    const codes = (this.state as unknown as { codes?: Texture }).codes;
    const model = (this.state as { topModel?: { shaderInputs: { setProps(p: object): void } } }).topModel;
    if (!codes || !model) return;
    model.shaderInputs.setProps({
      a5colour: { weights: c.weights, offset: c.offset, decode: c.decode ? 1 : 0, texels: t.texels, perRow: t.perRow, a5codes: codes },
    });
    super.draw(opts);
  }

  finalizeState(context: Parameters<SolidPolygonLayer["finalizeState"]>[0]) {
    super.finalizeState(context);
    (this.state as unknown as { codes?: Texture }).codes?.destroy();
  }
}
