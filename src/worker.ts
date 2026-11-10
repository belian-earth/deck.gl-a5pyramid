// Worker entry: opens shards from footer bytes and reads tiles. Every message
// carries an id; replies echo it with a result or an error. The work is
// handleRequest (tile-core.ts), which tests call in-process.

import { handleRequest, type WorkerRequest } from "./tile-core.js";

export type { WorkerRequest };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const { reply, transfer } = await handleRequest(ev.data);
  ctx.postMessage(reply, transfer);
};
