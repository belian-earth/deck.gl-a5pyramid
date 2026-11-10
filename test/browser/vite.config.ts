import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// the test page, with the fixture sets as static files (sirv serves byte ranges)
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  publicDir: fileURLToPath(new URL("../fixtures", import.meta.url)),
  server: { port: 5199, strictPort: true },
  worker: { format: "es" },
});
