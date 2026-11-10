// Test support: the fixture sets served by a stub fetch (with byte ranges,
// as a CDN serves them), and a worker that runs in-process.

import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { handleRequest, type WorkerRequest } from "../src/tile-core.js";
import type { SetIndex } from "../src/format.js";

export const ORIGIN = "http://fixtures.test/";
const DIR = fileURLToPath(new URL("./fixtures/", import.meta.url));

/** Requests served, for tests that count them. */
export const served: { path: string; range: string | null }[] = [];

/** Serve test/fixtures at ORIGIN: whole files, or one byte range (206). */
export function installFetch() {
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.origin + "/" !== ORIGIN) return new Response("not found", { status: 404 });
    const path = DIR + decodeURIComponent(url.pathname.slice(1));
    let body: Buffer;
    try {
      if (!statSync(path).isFile()) throw new Error("not a file");
      body = readFileSync(path);
    } catch {
      return new Response("not found", { status: 404 });
    }
    const range = new Headers(init?.headers).get("range");
    served.push({ path: url.pathname, range });
    const m = /^bytes=(\d+)-(\d+)$/.exec(range ?? "");
    if (!m) return new Response(new Uint8Array(body), { status: 200 });
    const start = Number(m[1]), end = Math.min(Number(m[2]), body.length - 1);
    return new Response(new Uint8Array(body.subarray(start, end + 1)), {
      status: 206,
      headers: { "content-range": `bytes ${start}-${end}/${body.length}` },
    });
  }) as typeof fetch;
}

/** A worker running handleRequest in-process, replying asynchronously. */
export class LocalWorker {
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  postMessage(m: WorkerRequest) {
    void handleRequest(m).then(({ reply }) => this.onmessage?.({ data: reply }));
  }
  terminate() {}
}
export const localWorker = () => new LocalWorker() as unknown as Worker;

export function readSet(name: string): SetIndex {
  return JSON.parse(readFileSync(DIR + name + "/set.json", "utf8"));
}

/** A shard's footer, as the source fetches it. */
export function footerOf(name: string, shard: { path: string; bytes: number; footer_bytes: number }): ArrayBuffer {
  const b = readFileSync(DIR + name + "/" + shard.path);
  return new Uint8Array(b.subarray(shard.bytes - shard.footer_bytes)).buffer;
}

/** The AEF decode, as the shader computes it. */
export function decode(code: number): number {
  const x = code / 127.5;
  return Math.sign(x) * x * x;
}
