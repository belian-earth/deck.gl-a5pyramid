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
  // inside the mask, the colours are the unmasked layer's
  const plain = await page.evaluate((n) => (window as any).masked(n, false), "store-v4");
  expect(plain.right).toBeGreaterThan(50);
  const keys = Object.keys(r.colours).filter((k) => k in plain.colours);
  const worst = Math.max(...keys.map((k) => Math.max(...r.colours[k].map((v: number, i: number) => Math.abs(v - plain.colours[k][i])))));
  const distinct = new Set(keys.map((k) => r.colours[k].slice(0, 3).join())).size;
  console.log(JSON.stringify({ worst, n: keys.length, distinct, first: keys.slice(0, 3).map((k) => [r.colours[k], plain.colours[k]]) }));
  expect(keys.length).toBeGreaterThan(50);
  expect(distinct).toBeGreaterThan(10);
  expect(worst).toBeLessThanOrEqual(1);
});

// A layer that is not pickable draws the same colours: the cells find their
// codes by deck.gl's picking index, which deck leaves constant without a
// picking buffer (every cell took one colour before ROW_INDEXES).
test("store-v4: the same colours when not pickable", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).masked === "function");
  const plain = await page.evaluate((n) => (window as any).masked(n, false, true), "store-v4");
  const unpickable = await page.evaluate((n) => (window as any).masked(n, false, false), "store-v4");
  const keys = Object.keys(plain.colours).filter((k) => k in unpickable.colours);
  const distinct = new Set(keys.map((k) => unpickable.colours[k].slice(0, 3).join())).size;
  const worst = Math.max(...keys.map((k) => Math.max(...plain.colours[k].map((v: number, i: number) => Math.abs(v - unpickable.colours[k][i])))));
  console.log(JSON.stringify({ n: keys.length, distinct, worst }));
  expect(keys.length).toBeGreaterThan(50);
  expect(distinct).toBeGreaterThan(10);
  expect(worst).toBeLessThanOrEqual(1);
});
