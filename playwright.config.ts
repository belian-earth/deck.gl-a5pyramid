import { defineConfig } from "@playwright/test";

// The installed Chrome (no browser download), headless, WebGL 2 in software.
export default defineConfig({
  testDir: "test/browser",
  testMatch: "*.e2e.ts",
  use: {
    baseURL: "http://127.0.0.1:5199",
    channel: "chrome",
    viewport: { width: 800, height: 600 },
    launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] },
  },
  webServer: {
    command: "npx vite --config test/browser/vite.config.ts --host 127.0.0.1",
    url: "http://127.0.0.1:5199",
    reuseExistingServer: false,
  },
});
