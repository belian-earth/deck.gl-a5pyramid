// GPU colours against the colourings' formulas, in Chrome: the drawn pixel
// at each sampled cell's centre within 1/255 of the CPU's value.

import { expect, test } from "@playwright/test";

for (const name of ["bands", "vector", "vector-rg", "store-v4"]) {
  test(`${name}: drawn colours match the formula`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => m.type() === "error" && !m.text().startsWith("Failed to load resource") && errors.push(m.text()));
    // a missing file other than the favicon
    page.on("response", (r) => r.status() >= 400 && !r.url().endsWith("/favicon.ico") && errors.push(`${r.status()} ${r.url()}`));
    await page.goto("/");
    await page.waitForFunction(() => typeof (window as any).check === "function");
    const r = await page.evaluate((n) => (window as any).check(n), name);
    console.log(JSON.stringify(r));
    expect(errors).toEqual([]);
    expect(r.tiles).toBeGreaterThan(0);
    expect(r.cells).toBeGreaterThan(20);
    expect(r.worst).toBeLessThanOrEqual(1);
  });
}

// A MaskExtension on the layer reaches its cell layers: nothing is drawn
// outside the mask.
test("store-v4: drawn only inside a mask", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && !m.text().startsWith("Failed to load resource") && errors.push(m.text()));
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).masked === "function");
  const r = await page.evaluate((n) => (window as any).masked(n), "store-v4");
  console.log(JSON.stringify(r));
  expect(errors).toEqual([]);
  expect(r.tiles).toBeGreaterThan(0);
  expect(r.left).toBeGreaterThan(50);
  expect(r.right).toBe(0);
});
