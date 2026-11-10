// deck.gl-a5pyramid: draw a5pyramid sets (A5 cells in Parquet, read with
// HTTP range requests) in deck.gl. The public surface; everything else is
// internal and may change.

export { PyramidSource, defaultWorker, type WorkerFactory, type SourceStats } from "./source";
export {
  A5PyramidLayer, emptyRenderStats, isSettled,
  type A5PyramidLayerProps, type RenderStats, type CellInfo,
} from "./layer";
export { DEFAULT_BUDGET, BASE_BUDGET, chooseLevel, type Budget, type Cost, type LevelChoice } from "./budget";
export {
  bandColouring, neutralColouring, colouring, weightIndex, decodeAef, WEIGHT_VEC4S,
  type Colouring, type Stretch,
} from "./colour";
export { ScreenPca, pcaBasis, pcaColouring, scores, covariance, type PcaBasis, type PcaView } from "./pca";
export { a5Resolution, cellPixels, cellSize, metresPerPixel, pickLod, viewportBox } from "./tiling";
export { FORMAT_VERSION, type Box, type SetIndex, type ShardEntry, type ColumnMeta, type Tier } from "./format";
export { AEF_DECODE, MAX_INPUTS, type PackedTile, type TileStats } from "./tile-core";
