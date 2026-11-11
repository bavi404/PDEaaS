/**
 * Documentation screenshots of the working interface (real backend + checkpoint).
 * Run explicitly:  SCREENSHOTS=1 npx playwright test e2e/screenshots.spec.ts --trace off
 * Output: ../docs/screenshots/<project>-<page>.png (desktop at 2x device scale).
 */
import { expect, test } from "@playwright/test";

test.skip(!process.env.SCREENSHOTS, "set SCREENSHOTS=1 to regenerate documentation screenshots");
test.use({ deviceScaleFactor: 2 });

const out = (project: string, name: string) => `../docs/screenshots/${project}-${name}.png`;

test("capture", async ({ page }, info) => {
  test.setTimeout(180_000);
  const p = info.project.name;
  // a benchmark and a custom prediction so the dashboard shows real activity
  await page.request.post("/api/predict", { data: { sample_id: "test-8142", resolution: 64 } });
  await page.request.post("/api/predict", { data: { sample_id: "test-4701", resolution: 128 } });

  await page.goto("/workspace");
  await page.getByTestId("example-test-2113").click();
  await expect(page.getByTestId("hm-workspace-input").locator(".main-svg").first()).toBeVisible();
  await page.waitForTimeout(600);
  await page.screenshot({ path: out(p, "2-workspace-benchmark"), fullPage: true });

  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");
  await expect(page.getByTestId("hm-error-ref").locator(".main-svg").first()).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({ path: out(p, "3-results-benchmark"), fullPage: true });

  await page.goto("/workspace");
  await page.getByText("Custom field", { exact: true }).click();
  await page.getByTestId("gen-seed").fill("11");
  await page.getByTestId("gen-high").fill("2");
  await expect(page.getByTestId("ood-preview")).toBeVisible();
  await page.waitForTimeout(600);
  await page.screenshot({ path: out(p, "4-workspace-custom"), fullPage: true });
  await page.getByTestId("predict-button").click();
  await page.waitForURL("**/results");
  await expect(page.getByTestId("hm-prediction").locator(".main-svg").first()).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({ path: out(p, "5-results-custom"), fullPage: true });

  await page.goto("/");
  await expect(page.getByTestId("activity-table")).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot({ path: out(p, "1-dashboard"), fullPage: true });

  await page.goto("/model");
  await expect(page.getByTestId("model-details")).toBeVisible();
  await page.screenshot({ path: out(p, "6-model"), fullPage: true });

  await page.goto(process.env.E2E_DEAD_URL ?? "http://127.0.0.1:3101/");
  await expect(page.getByTestId("error-state").first()).toBeVisible();
  await page.screenshot({ path: out(p, "7-backend-unavailable"), fullPage: false });
});
