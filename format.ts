// The a5pyramid set format (a5pyramid FORMAT.md, "Pyramid sets", version 2):
// set.json and each shard's index, as this reader takes them.

/** `[west, south, east, north]` in degrees; `west > east` crosses the antimeridian. */
export type Box = [number, number, number, number];

export interface Tier {
  lods: [number, number];
  shard_resolution: number;
  tile_size: number;
}

export interface ShardEntry {
  tier: number;
  cell: string;
  path: string;
  rows: number;
  tiles: number;
  bytes: number;
  footer_bytes: number;
  bbox: Box;
}

export interface ColumnMeta {
  name: string;
  type: string;
  reducer: string;
}

export interface SetIndex {
  format: string;
  format_version: number;
  generator: string;
  a5_crate: string;
  a5_index: string;
  data_resolution: number;
  columns: ColumnMeta[];
  tiers: Tier[];
  bbox: Box;
  shards: ShardEntry[];
  metadata: Record<string, string>;
}

export interface LevelMeta {
  lod: number;
  rows: number;
  tiles: [number, number];
}

/** A shard's (or a single file's) index, the `a5pyramid` key-value entry. */
export interface PyramidIndex {
  format_version: number;
  data_resolution: number;
  tile_size: number;
  bbox: Box;
  columns: ColumnMeta[];
  levels: LevelMeta[];
  tile_bbox: number[];
  shard?: { tier: number; cell: string; resolution: number };
}

/** The newest format version this reader reads, and the A5 index of a5-js 1.x. */
export const FORMAT_VERSION = 2;
export const A5_INDEX = "1.0";

export function checkSet(s: SetIndex): SetIndex {
  if (s.format !== "a5pyramid-set") throw new Error(`not an a5pyramid set: format ${JSON.stringify(s.format)}`);
  if (s.format_version > FORMAT_VERSION) throw new Error(`set format version ${s.format_version} is newer than ${FORMAT_VERSION}`);
  if (s.a5_index !== A5_INDEX) throw new Error(`set holds A5 index ${s.a5_index} ids; this reader draws ${A5_INDEX}`);
  return s;
}

export function checkIndex(i: PyramidIndex, a5Index: string | undefined, where: string): PyramidIndex {
  if (i.format_version > FORMAT_VERSION) throw new Error(`${where}: format version ${i.format_version} is newer than ${FORMAT_VERSION}`);
  if (a5Index !== A5_INDEX) throw new Error(`${where}: A5 index ${a5Index}, not ${A5_INDEX}`);
  return i;
}

/** The tier holding `lod`, or -1. */
export function tierOf(set: SetIndex, lod: number): number {
  return set.tiers.findIndex((t) => lod >= t.lods[0] && lod <= t.lods[1]);
}

/** Every LOD the tiers hold, ascending. */
export function setLods(set: SetIndex): number[] {
  const out: number[] = [];
  for (const t of set.tiers) for (let l = t.lods[0]; l <= t.lods[1]; l++) out.push(l);
  return out.sort((a, b) => a - b);
}

function intervals(west: number, east: number): [number, number][] {
  return west > east ? [[west, 180], [-180, east]] : [[west, east]];
}

/** Whether two boxes meet, each split at the antimeridian where it crosses it. */
export function boxesMeet(a: Box, b: Box): boolean {
  if (a[1] > b[3] || b[1] > a[3]) return false;
  for (const [w1, e1] of intervals(a[0], a[2])) {
    for (const [w2, e2] of intervals(b[0], b[2])) if (w1 <= e2 && w2 <= e1) return true;
  }
  return false;
}

export function tileBox(index: PyramidIndex, i: number): Box {
  const b = index.tile_bbox;
  return [b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]];
}
