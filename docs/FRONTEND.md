# PDEFlow frontend (M5)

The frontend is a Next.js 16 app (App Router), written in TypeScript, with Plotly heatmaps. It is in `frontend/`.

- **Real data only.** Every number and field it shows comes from the running FastAPI backend. There are no mock predictions and no hard-coded metrics.
- **Fixtures stay in tests.** Mocks and fixtures exist only in the unit tests.

## 1. Run it

```bash
# backend (from the repo root)
.venv/bin/uvicorn pdeflow.api.app:app --app-dir backend --host 127.0.0.1 --port 8000
```

```bash
# frontend
cd frontend
npm ci
npm run build
PORT=3000 PDEFLOW_API_URL=http://127.0.0.1:8000 node .next/standalone/server.js
```

Development: `npm run dev` (port 3000), with `PDEFLOW_API_URL` defaulting to `http://127.0.0.1:8000`.

| Command | What it does |
|---|---|
| `npm run build` | production build (`output: "standalone"`) and a copy of the static assets |
| `npm run typecheck` | `tsc --noEmit`, strict mode |
| `npm test` | Vitest + React Testing Library: 52 tests |
| `npm run e2e` | Playwright against the **real stack**: 26 tests (13 desktop + 13 mobile). Run `npm run build` first. |
| `SCREENSHOTS=1 npx playwright test e2e/screenshots.spec.ts --trace off` | regenerates `docs/screenshots/` |

## 2. Architecture

```
browser ──/api/*──▶ Next.js route handler (src/app/api/[...path]/route.ts) ──▶ FastAPI (PDEFLOW_API_URL)
```

| Decision | Alternatives | Justification |
|---|---|---|
| **Same-origin proxy route handler** | `next.config` rewrites; browser calls FastAPI directly | Rewrites are fixed at build time. The handler reads `PDEFLOW_API_URL` at run time, so one image works locally and on HF Spaces. It allows only the documented endpoints, adds a 30 s timeout, and returns a clear **502 `{detail}`** when the backend is down. Same origin means no CORS in production. |
| Client components with explicit loading / success / error states (`useResource`) | server components that fetch at render | The pages show live service state (activity, health). A failed backend must show a retryable error state, not a server-render failure. |
| **plotly.js-dist-min**, loaded dynamically inside one `Heatmap` component | react-plotly.js; Recharts / visx | Plotly gives a real heatmap with hover, zoom, colour bars and PNG export. The react-plotly wrapper lags React 19; a 30-line wrapper with `Plotly.react` / `purge` avoids that. Dynamic import keeps it out of server rendering. |
| Plain CSS with design tokens (`globals.css`) | Tailwind, component libraries | No extra build dependency and a small, auditable stylesheet. One restrained palette: steel-blue accent; violet tint reserved for *secondary* (reference) data. |
| Last prediction kept in React context + `sessionStorage` | URL state; server storage | A 128×128 result is about 330 kB, too big for a URL. sessionStorage survives a reload of the Results page and stays in the user's tab. |
| Pinned dependency versions, each at least 2 weeks old; `.npmrc legacy-peer-deps=true` | npm default peer resolution | Vite 8's optional `@vitejs/devtools-*` peers form a cycle that crashes the npm 10 resolver ("edgesOut" of null). All required peers (`vite`, `@testing-library/dom`, React) are pinned explicitly. `npm ci` from the committed lockfile is verified. |

**Source layout** (`src/`):
- `lib/api.ts`: typed client and response types mirroring `backend/pdeflow/api/schemas.py`. `ApiError` kinds: `unavailable`, `validation`, `not_found`, `too_large`, `server`, `malformed`.
- `lib/fields.ts`:
  - custom-field parsing (JSON, or CSV/whitespace rows)
  - validation that mirrors the API's checks: 64/128 square, finite, > 0, ≤ 1e6
  - distribution-shift warnings with the API's wording
  - the seeded two-phase field generator
- `components/`: reusable building blocks:
  - `Heatmap`
  - `ResultsView`
  - `MetricsTable`
  - `ActivityList`
  - `AppShell`
  - `ui.tsx` (Card, Stat, Notice, Loading/Error/Empty states, Segmented, KeyValue)
  - `workspace/ExamplePicker`, `workspace/CustomFieldEditor`
- `app/`: the pages `/`, `/workspace`, `/results`, `/model`, and the `/api/[...path]` proxy.

## 3. Pages

| Page | Content | API calls |
|---|---|---|
| **Dashboard** `/` | introduction, PDE statement, model overview, held-out test performance (official target vs reference, in separate columns), recent prediction activity (polled every 15 s), API status with checkpoint hash | `/model`, `/activity`, `/health` |
| **Solver workspace** `/workspace` | PDE (Darcy, the only operator served); input type; benchmark example picker (24 held-out test samples) **or** custom field (seeded generator / paste / upload); resolution 64 or 128; 128×128 inference path; input heatmap; pre-submission validation and out-of-distribution warnings | `/samples`, `/samples/{id}`, `/predict` |
| **Results** `/results` | see §4 | `/model` (service version) |
| **Model** `/model` | FNO architecture (modes, hidden channels, layers, parameters), training data and setup, normalisation, evaluation table, target definitions, limitations, checkpoint provenance | `/model` |

The "recent activity" feed comes from the backend's `GET /activity`, added in M5. It is an in-memory ring buffer of the last 100 predictions, metadata only:
- time, input kind, sample id, resolution, strategy, latency, number of warnings
- rel-L2 vs official, for benchmark examples only

It resets when the API restarts and stores no input or output fields.

## 4. Scientific presentation rules (implemented and tested)

1. **Benchmark example** (bundled test sample):
   - **Comparison grid:** permeability | predicted pressure û | **official PDEBench target u** | |û − u|.
   - **Shared colour range:** û and u use the same zmin/zmax.
   - **Headline metric:** the primary metric is the relative L2 vs the **official target**.
2. **Independent steady-state reference:**
   - It appears only in a separate, violet-tinted card titled "Secondary analysis".
   - The card has its own heatmaps (u_ref, |û − u_ref|), rel-L2 vs the reference, and the official-vs-reference gap.
   - It never appears in the primary comparison; an e2e test checks this.
3. **Custom input:**
   - Shows the permeability and prediction only, plus a notice that no ground truth exists.
   - **No target, error map or error metric is shown**, so nothing is fabricated; unit and e2e tests check this.
4. **Out-of-distribution input:**
   - Values outside [0.1, 1.0], or non-binary fields, show a warning before submission (client check).
   - The API's own warnings are shown after prediction.
   - The dashboard activity marks such runs "OOD".
5. **Error metrics appear only when a valid reference exists:** `evaluation` and `targets` must both be present in the API response.
6. **Heatmaps:**
   - Physical cell-centre coordinates on both axes, with aspect ratio locked to 1.
   - Labelled colour bar; hover shows x, y and the value.
   - Colour maps: Viridis (pressure), Cividis (permeability, colour-blind safe), Inferno anchor colours (error).
   - API grids are `[x][y]` and are transposed for Plotly; this is unit tested.

## 5. Tests

**Unit tests:** Vitest + RTL, 52 tests, output in `reports/m5_vitest.txt`.

| File | Covers |
|---|---|
| `tests/api.test.ts` | API integration contract with a fake `fetch`:
  - URLs, JSON bodies
  - 422 detail formatting and request id
  - network failure and proxy 502 → *unavailable*
  - 404 / 413, non-JSON → *malformed* |
| `tests/fields.test.ts` | parsing JSON/CSV; 10 malformed-input cases; OOD warnings; generator determinism, binary values, fraction, 0%/100% |
| `tests/heatmap.test.tsx` | `[x][y]` → Plotly transpose, cell-centre coordinates, colour bar, hover template, axis titles, aspect lock, Plotly called |
| `tests/results.test.tsx` | benchmark layout + separate secondary section; custom input has no target/error/metric; OOD warning; no metrics without an evaluation |
| `tests/pages.test.tsx` | dashboard (metrics, activity, empty state, backend down + retry, metrics withheld when the report doesn't match); workspace (example selection, 64² switch, generated custom submit, OOD preview, malformed paste, CSV upload, server 422 shown with request id, examples unavailable); results empty / stored; model page |

**End-to-end tests:** Playwright (`e2e/app.spec.ts`), 26 tests, output in `reports/m5_e2e.txt`.
- **Setup:**
  - uvicorn serves the committed checkpoint, and the production Next.js server sits in front of it.
  - A **second Next.js server points at a closed port**, so the unavailable-backend state is real.
  - Every test runs on desktop (1440×900) and mobile (Pixel 7).
  - Every test fails on any browser console error or uncaught page error. The one exception is the deliberately aborted request in the failure test.
- **Coverage:**
  - The dashboard numbers equal `/api/model`.
  - Full benchmark workflow, 128² and 64²: the displayed rel-L2 equals the API's value for the same request, all heatmaps render with colour bars, and hover shows coordinates and value.
  - Custom workflow with no fabricated ground truth.
  - OOD warnings shown before and after prediction.
  - Malformed paste rejected, valid JSON upload accepted.
  - The proxy passes 422 through and returns 404 for unknown paths.
  - Activity feed, model page and results empty state.
  - Real backend outage, plus a failure during prediction.
  - No horizontal overflow on any page at either width.

## 6. Screenshots

These are in `docs/screenshots/`, captured from the running stack: desktop at 1440 px wide (2× scale), mobile on a Pixel 7.

| | Desktop | Mobile |
|---|---|---|
| Dashboard | `desktop-1-dashboard.png` | `mobile-1-dashboard.png` |
| Workspace, benchmark | `desktop-2-workspace-benchmark.png` | `mobile-2-workspace-benchmark.png` |
| Results, benchmark (official target + secondary reference) | `desktop-3-results-benchmark.png` | `mobile-3-results-benchmark.png` |
| Workspace, custom OOD field | `desktop-4-workspace-custom.png` | `mobile-4-workspace-custom.png` |
| Results, custom (no ground truth, OOD warning) | `desktop-5-results-custom.png` | `mobile-5-results-custom.png` |
| Model information | `desktop-6-model.png` | `mobile-6-model.png` |
| Backend unavailable | `desktop-7-backend-unavailable.png` | `mobile-7-backend-unavailable.png` |

## 7. Known limitations

- The PDE selector offers one operator (Darcy, β = 1), because the deployment serves exactly one trained checkpoint.
- The custom-field generator resembles, but is not drawn from, the PDEBench input distribution. The UI says so.
- Activity is per API process and in memory: it resets on restart and is shared by all users of one deployment. It holds metadata only.
- Plotly adds about 1.4 MB (minified) to the client bundle. It is loaded only when a heatmap is shown.
