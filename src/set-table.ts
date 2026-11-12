// A set's index as a Parquet table (a5pyramid format version 4, "Shard
// tables"): one row per shard, the set's own fields in the footer, boxes
// unwrapped. Read whole on the main thread: a year of a store is a few
// hundred rows.

import { parquetMetadata, parquetReadObjects, type Compressors } from "hyparquet";
import type { SetIndex, ShardEntry } from "./format.js";

/** The footer key of the set's fields. */
export const SET_META_KEY = "a5pyramid_set";
const COLUMNS = ["tier", "cell", "path", "rows", "tiles", "bytes", "footer_bytes", "west", "south", "east", "north"];

/** The set a shard table at `url` describes, its shards' boxes wrapped
 * (`west > east` across the antimeridian). */
export async function readSetTable(url: string, compressors?: Compressors): Promise<SetIndex> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  const metadata = parquetMetadata(buf);
  const kv = Object.fromEntries((metadata.key_value_metadata ?? []).map((e) => [e.key, e.value ?? ""]));
  if (!kv[SET_META_KEY]) throw new Error(`${url}: not an a5pyramid shard table (no ${SET_META_KEY})`);
  const set = JSON.parse(kv[SET_META_KEY]) as SetIndex;
  const file = { byteLength: buf.byteLength, slice: (a: number, b?: number) => buf.slice(a, b) };
  const rows = (await parquetReadObjects({ file, metadata, columns: COLUMNS, compressors })) as Record<string, unknown>[];
  const num = (v: unknown) => Number(v);
  set.shards = rows.map((r): ShardEntry => ({
    tier: num(r.tier),
    cell: BigInt.asUintN(64, BigInt(r.cell as bigint)).toString(16).padStart(16, "0"),
    path: (r.path as string | null) ?? null,
    rows: num(r.rows),
    tiles: num(r.tiles),
    bytes: num(r.bytes),
    footer_bytes: num(r.footer_bytes),
    bbox: r.west == null ? null : [num(r.west), num(r.south), num(r.east) > 180 ? num(r.east) - 360 : num(r.east), num(r.north)],
  }));
  return set;
}
