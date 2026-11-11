import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against the REAL stack: uvicorn serving the committed checkpoint + the production Next.js build.
 * Run `npm run build` first. Ports: API 8100, web 3100 (kept away from dev ports).
 */
const API_PORT = 8100;
const WEB_PORT = 3100;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${WEB_PORT}`, trace: "off" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: [
    {
      command: `../.venv/bin/uvicorn pdeflow.api.app:app --app-dir ../backend --host 127.0.0.1 --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/health`,
      env: { PDEFLOW_LOG_FILE: "../logs/m5_e2e_api.jsonl", PDEFLOW_CORS_ORIGINS: "" },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `node .next/standalone/server.js`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: { PORT: String(WEB_PORT), HOSTNAME: "127.0.0.1", PDEFLOW_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
