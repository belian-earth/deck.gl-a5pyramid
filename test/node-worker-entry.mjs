// A decode worker for Node's worker_threads (the benchmarks): the bundled
// worker's work on the built package. Run `npm run build` first.
import { parentPort } from "node:worker_threads";
import { ZSTDDecoder } from "zstddec";
import { handleRequest, setCompressors } from "../dist/tile-core.js";

const zstd = new ZSTDDecoder();
const ready = zstd.init().then(() => setCompressors({ ZSTD: (input, outputLength) => zstd.decode(input, outputLength) }));

parentPort.on("message", async (m) => {
  await ready;
  const { reply, transfer } = await handleRequest(m);
  parentPort.postMessage(reply, transfer);
});
