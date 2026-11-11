/**
 * Gradio UI of the ZeroGPU Space app (deploy/zerogpu/app.py), against a running integrated app:
 *   E2E_GRADIO=1 E2E_BASE_URL=http://127.0.0.1:7860 npx playwright test e2e/gradio.spec.ts --trace off
 * Skipped unless E2E_GRADIO=1 (the Next.js/Docker stack has no /gradio).
 */
import { expect, test } from "@playwright/test";

test.skip(!process.env.E2E_GRADIO, "set E2E_GRADIO=1 and E2E_BASE_URL to the ZeroGPU Space app");

const pct = (x: number) => `${(x * 100).toFixed(2)}%`;

test("Gradio benchmark example shows the same metric as /api/predict, with attribution", async ({ page, request }) => {
  const api = await (await request.post("/api/predict", { data: { sample_id: "test-4701", resolution: 128 } })).json();
  const want = pct(api.evaluation.primary_vs_official_pdebench_target.relative_l2);
  await page.goto("/gradio/");
  await expect(page.getByText("CC BY 4.0").first()).toBeVisible();
  await expect(page.getByText("doi:10.18419/darus-2986").first()).toBeVisible();
  await page.getByRole("button", { name: "Run prediction" }).first().click();
  await expect(page.getByText(`Relative L2 vs official PDEBench target: ${want}`)).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("img[src*='data:image'], img[src*='/file'], .js-plotly-plot, canvas").first()).toBeVisible();
});

test("Gradio custom field: prediction without ground truth, OOD warning", async ({ page }) => {
  await page.goto("/gradio/");
  await page.getByRole("tab", { name: "Custom field" }).click();
  const grid = JSON.stringify(Array.from({ length: 64 }, () => Array(64).fill(0.5)));
  await page.getByLabel("…or paste here").fill(grid);
  await page.getByRole("button", { name: "Run prediction" }).last().click();
  await expect(page.getByText("No ground truth exists for custom inputs")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("non-binary field").first()).toBeVisible();
});

test("Gradio reports the API's validation error for a malformed field", async ({ page }) => {
  await page.goto("/gradio/");
  await page.getByRole("tab", { name: "Custom field" }).click();
  await page.getByLabel("…or paste here").fill("[[1, 2], [3, 4]]");
  await page.getByRole("button", { name: "Run prediction" }).last().click();
  await expect(page.getByText(/64x64 or 128x128/).first()).toBeVisible({ timeout: 60_000 });
});
