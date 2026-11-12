// Worker entry: opens shards from footer bytes and reads tiles. Every message
// carries an id; replies echo it with a result or an error. The work is
// handleRequest (tile-core.ts), which tests call in-process. Zstandard pages
// (the a5embeddings store's) are decoded by zstddec's wasm build, set up
// before the first message is handled.

import { ZSTDDecoder } from "zstddec";
import { handleRequest, setCompressors, type WorkerRequest } from "./tile-core.js";

export type { WorkerRequest };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

const zstd = new ZSTDDecoder();
const ready = zstd.init().then(
  () => setCompressors({ ZSTD: (input: Uint8Array, outputLength: number) => zstd.decode(input, outputLength) }),
  (e) => console.error(`[deck.gl-a5pyramid] zstd decoder: ${e}`),
);

ctx.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  await ready;
  const { reply, transfer } = await handleRequest(ev.data);
  ctx.postMessage(reply, transfer);
};
