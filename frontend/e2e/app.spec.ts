/**
 * End-to-end tests against the real stack (uvicorn + committed checkpoint, production Next.js server).
 * Every test fails on any browser console error or uncaught page error (except where an error is the point).
 */
import { expect, test as base, type Page } from "@playwright/test";

const DEAD = "http://127.0.0.1:3101"; // web server whose backend URL points at a closed port

const test = base.extend<{ consoleErrors: string[] }>({
  consoleErrors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    await use(errors);
  },
});

const pct = (x: number) => `${(x * 100).toFixed(2)}%`;

async function plotRendered(page: Page, testId: string) {
  await expect(page.getByTestId(testId).locator(".main-svg").first()).toBeVisible();
  await expect(page.getByTestId(testId).locator(".colorbar")).toHaveCount(1);
}

async function noHorizontalOverflow(page: Page) {
  const { sw, cw } = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
  expect(sw, "page must not scroll horizontally").toBeLessThanOrEqual(cw + 1);
}

test("dashboard shows the API's model card, metrics and status", async ({ page, request, consoleErrors }) => {
  const card = await (await request.get("/api/model")).json();
  await page.goto("/");
  await expect(page.getByTestId("stat-test-64")).toContainText(pct(card.test_metrics.res64_fno_native.primary_vs_official_pdebench_target.mean));
  await expect(page.getByTestId("stat-test-128")).toContainText(pct(card.test_metrics.res128_fno64_interpolated.primary_vs_official_pdebench_target.mean));
  await expect(page.getByTestId("metrics-table")).toContainText(pct(card.test_metrics.res128_fno128_native.primary_vs_official_pdebench_target.mean));
  await expect(page.getByTestId("api-status")).toContainText("API online");
  await expect(page.getByTestId("api-status")).toContainText(card.checkpoint.sha256.slice(0, 12));
  await noHorizontalOverflow(page);
  expect(consoleErrors).toEqual([]);
});

test("benchmark example: select, predict, compare with the official target", async ({ page, request, consoleErrors, isMobile }) => {
  await page.goto("/workspace");
  await plotRendered(page, "hm-workspace-input");
  await page.getByTestId("example-test-4701").click();
  await expect(page.getByTestId("example-test-4701")).toHaveAttribute("aria-selected", "true");
  await expect(page.getByTestId("input-preview")).toContainText("#4701");
  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");

  const view = page.getByTestId("results-view");
  await expect(view).toHaveAttribute("data-kind", "benchmark_example");
  for (const id of ["hm-input", "hm-prediction", "hm-official", "hm-error"]) await plotRendered(page, id);

  // the displayed metric equals what the API computes for the same request
  const api = await (await request.post("/api/predict", { data: { sample_id: "test-4701", resolution: 128, strategy: "fno64_interpolated" } })).json();
  await expect(page.getByTestId("stat-primary")).toContainText(pct(api.evaluation.primary_vs_official_pdebench_target.relative_l2));

  // secondary analysis is separate and labelled as such
  const sec = page.getByTestId("secondary-analysis");
  await expect(sec).toContainText("Secondary analysis: independent steady-state reference");
  await expect(sec.getByTestId("stat-secondary")).toContainText(pct(api.evaluation.secondary_vs_independent_steady_state_reference.relative_l2));
  await plotRendered(page, "hm-reference");
  await expect(page.locator("section", { hasText: "Comparison with the official PDEBench target" }).getByTestId("hm-reference")).toHaveCount(0);

  // hover shows coordinates and value
  const drag = page.getByTestId("hm-prediction").locator(".nsewdrag").first();
  await drag.scrollIntoViewIfNeeded();
  const box = await drag.boundingBox();
  expect(box).not.toBeNull();
  if (isMobile) await page.touchscreen.tap(box!.x + box!.width / 2, box!.y + box!.height / 2);
  else await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await expect(page.locator(".hoverlayer .hovertext").first()).toContainText("û =");
  await expect(page.locator(".hoverlayer .hovertext").first()).toContainText("x = 0.");

  await noHorizontalOverflow(page);
  expect(consoleErrors).toEqual([]);
});

test("benchmark example at 64x64 runs at the training resolution", async ({ page, consoleErrors }) => {
  await page.goto("/workspace");
  await plotRendered(page, "hm-workspace-input");
  await page.getByText("64 × 64", { exact: true }).click();
  await expect(page.getByTestId("hm-workspace-input")).toContainText("64×64");
  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");
  await expect(page.getByTestId("results-view")).toContainText("64×64");
  await expect(page.getByTestId("results-view")).toContainText("fno_native");
  await plotRendered(page, "hm-error");
  expect(consoleErrors).toEqual([]);
});

test("custom field: prediction without fabricated ground truth or metrics", async ({ page, consoleErrors }) => {
  await page.goto("/workspace");
  await page.getByText("Custom field", { exact: true }).click();
  await plotRendered(page, "hm-workspace-input");
  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");
  const view = page.getByTestId("results-view");
  await expect(view).toHaveAttribute("data-kind", "custom");
  await plotRendered(page, "hm-prediction");
  for (const id of ["hm-official", "hm-error", "hm-reference", "stat-primary", "secondary-analysis"]) await expect(page.getByTestId(id)).toHaveCount(0);
  await expect(page.getByTestId("no-ground-truth")).toBeVisible();
  await expect(view).not.toContainText("Relative L2");
  await noHorizontalOverflow(page);
  expect(consoleErrors).toEqual([]);
});

test("out-of-distribution custom input is flagged before and after prediction", async ({ page, consoleErrors }) => {
  await page.goto("/workspace");
  await page.getByText("Custom field", { exact: true }).click();
  await page.getByTestId("gen-high").fill("3");
  await expect(page.getByTestId("ood-preview")).toContainText("outside the training range");
  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");
  await expect(page.getByTestId("ood-warning")).toContainText("values outside the training range [0.1, 1.0]");
  expect(consoleErrors).toEqual([]);
});

test("malformed custom input is rejected; a valid upload is accepted", async ({ page, consoleErrors }) => {
  await page.goto("/workspace");
  await page.getByText("Custom field", { exact: true }).click();
  await page.getByText("Paste or upload", { exact: true }).click();
  await page.getByTestId("paste-input").fill("[[0.1, 1.0], [1.0, 0.1]]");
  await expect(page.getByTestId("paste-error")).toContainText("Size 2×2 is not supported");
  await expect(page.getByTestId("predict-button")).toBeDisabled();
  await page.getByTestId("paste-input").fill("[[0.1, -1]]");
  await expect(page.getByTestId("paste-error")).toBeVisible();

  const field = Array.from({ length: 64 }, (_, i) => Array.from({ length: 64 }, (_, j) => ((i >> 3) + (j >> 3)) % 2 ? 1 : 0.1));
  await page.getByTestId("file-input").setInputFiles({ name: "field.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(field)) });
  await expect(page.getByTestId("paste-ok")).toContainText("64×64");
  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");
  await expect(page.getByTestId("results-view")).toHaveAttribute("data-kind", "custom");
  expect(consoleErrors).toEqual([]);
});

test("the proxy passes the API's validation errors through", async ({ request }) => {
  const r = await request.post("/api/predict", { data: { permeability: [[1, 2], [3, 4]] } });
  expect(r.status()).toBe(422);
  const b = await r.json();
  expect(b.detail).toContain("64x64 or 128x128");
  expect(b.request_id).toBeTruthy();
  expect((await request.get("/api/not-an-endpoint")).status()).toBe(404);
});

test("recent activity on the dashboard reflects served predictions", async ({ page, request, consoleErrors }) => {
  await request.post("/api/predict", { data: { sample_id: "test-2113", resolution: 64 } });
  await page.goto("/");
  const table = page.getByTestId("activity-table");
  await expect(table).toBeVisible();
  await expect(table).toContainText("test-2113");
  await expect(table).toContainText("Benchmark");
  expect(consoleErrors).toEqual([]);
});

test("model page shows architecture, training data, evaluation and limitations", async ({ page, request, consoleErrors }) => {
  const card = await (await request.get("/api/model")).json();
  await page.goto("/model");
  await expect(page.getByTestId("model-modes")).toHaveText(card.checkpoint.model_config.n_modes.join(" × "));
  await expect(page.getByTestId("model-hidden")).toHaveText(String(card.checkpoint.model_config.hidden_channels));
  await expect(page.getByTestId("model-layers")).toHaveText(String(card.checkpoint.model_config.n_layers));
  await expect(page.getByTestId("model-sha")).toHaveText(card.checkpoint.sha256);
  await expect(page.getByTestId("evaluation-card")).toContainText(pct(card.test_metrics.res64_fno_native.secondary_vs_independent_steady_state_reference.mean));
  await expect(page.getByTestId("limitations")).toContainText("binary permeability fields");
  await noHorizontalOverflow(page);
  expect(consoleErrors).toEqual([]);
});

test("results page has an empty state before any prediction", async ({ page, consoleErrors }) => {
  await page.goto("/results");
  await expect(page.getByTestId("empty-state")).toContainText("No prediction yet");
  expect(consoleErrors).toEqual([]);
});

test.describe("unavailable backend (real: web server pointed at a closed port)", () => {
  test("dashboard, workspace and model page show error states", async ({ page }) => {
    const r = await page.request.get(`${DEAD}/api/health`);
    expect(r.status()).toBe(502);
    expect((await r.json()).detail).toContain("unavailable");

    await page.goto(`${DEAD}/`);
    await expect(page.getByTestId("api-status")).toContainText("API unavailable");
    await expect(page.getByTestId("error-state").first()).toContainText("API unavailable");
    await page.goto(`${DEAD}/workspace`);
    await expect(page.getByTestId("error-state").first()).toContainText("API unavailable");
    await expect(page.getByTestId("predict-button")).toBeDisabled();
    await page.goto(`${DEAD}/model`);
    await expect(page.getByTestId("error-state")).toContainText("Retry");
    await noHorizontalOverflow(page);
  });

  test("a backend failure during prediction is reported in the workspace", async ({ page, consoleErrors }) => {
    await page.goto("/workspace");
    await plotRendered(page, "hm-workspace-input");
    await page.route("**/api/predict", (route) => route.abort("connectionrefused"));
    await page.getByTestId("predict-button").click();
    await expect(page.getByTestId("submit-error")).toContainText("API unavailable");
    expect(page.url()).toContain("/workspace");
    // the aborted request is the only expected console error
    expect(consoleErrors.filter((e) => !e.includes("ERR_CONNECTION_REFUSED") && !e.includes("Failed to load resource"))).toEqual([]);
  });
});

test("all pages fit the viewport without horizontal scrolling", async ({ page, consoleErrors }) => {
  for (const path of ["/", "/workspace", "/model", "/results"]) {
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    await noHorizontalOverflow(page);
  }
  expect(consoleErrors).toEqual([]);
});
