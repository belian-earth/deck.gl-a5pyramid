// Worker entry: opens shards from footer bytes and reads tiles. Every message
// carries an id; replies echo it with a result or an error.

import { keyValues, openShard, hasShard, readTile, transferables } from "./tile-core";

export type WorkerRequest =
  | { id: number; type: "index"; url: string; bytes: number; footer: ArrayBuffer }
  | {
      id: number; type: "decode"; url: string; bytes: number; footer?: ArrayBuffer; tile: number; lod: number;
      columns: string[]; stats?: boolean; vector?: { name: string; dim: number } | null;
    };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const m = ev.data;
  try {
    if (m.type === "index") {
      const md = openShard(m.url, m.bytes, m.footer);
      ctx.postMessage({ id: m.id, kv: keyValues(md), rowGroups: md.row_groups.length });
      return;
    }
    if (!hasShard(m.url)) {
      if (!m.footer) throw new Error(`no footer for ${m.url}`);
      openShard(m.url, m.bytes, m.footer);
    }
    const tile = await readTile(m.url, m.tile, m.lod, m.columns, m.stats ?? false, m.vector ?? null);
    ctx.postMessage({ id: m.id, tile }, transferables(tile));
  } catch (e) {
    ctx.postMessage({ id: m.id, error: String((e as Error)?.message ?? e) });
  }
};
