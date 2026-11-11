# PDEFlow API (M4)

This is a FastAPI service that serves the **evaluated** FNO checkpoint (docs/RESULTS.md). The code lives in:
- `backend/pdeflow/api/` (app, schemas, settings)
- `backend/pdeflow/inference.py` (inference path)

Run it locally:

```bash
.venv/bin/uvicorn pdeflow.api.app:app --app-dir backend --host 127.0.0.1 --port 8000
```

Interactive docs are at `/docs` and the schema at `/openapi.json`.

## 1. The service serves exactly the evaluated model

| Check | Where |
|---|---|
| Checkpoint SHA-256 is pinned (`d7133c7c…124df7`, the same as `reports/m2_eval_fno_baseline.json`); startup **fails** if the file differs | `pdeflow/api/settings.py`, `Predictor.__init__`; test `test_startup_refuses_wrong_checkpoint_hash` |
| The API's inference path, run on all 1,000 test samples, reproduces the evaluation report **bit-for-bit** (max relative difference 0) for all three strategies | `scripts/verify_api_inference.py` → `reports/m4_inference_equivalence.json` |
| Inference reuses the validated code: `load_checkpoint`, `predict`, the train-only normalisers stored in the checkpoint, `subsample`, `cell_centres` and `linear_upsample` | `pdeflow/inference.py` |
| `/model` shows test metrics only if the report's checkpoint hash equals the served hash | `_load_metrics`; test `test_model_card_omits_metrics_from_a_different_checkpoint` |

**Batch-size effect:** a single-sample request can differ from the batched evaluation by float32 rounding (measured max |Δu| 3.0e-7 for test idx 4701). The reason is that CPU kernels round differently depending on batch size. With the same batch composition, results are bit-identical.

## 2. Two targets, never mixed

Bundled samples carry two targets in separately named fields:
- `official_pdebench_target`: the PRIMARY benchmark, which the model was trained and selected on.
- `independent_steady_state_reference`: SECONDARY analysis only.

`/predict` on a sample returns, each under its own key:
- `primary_vs_official_pdebench_target`
- `secondary_vs_independent_steady_state_reference`
- `official_target_vs_reference`, the gap between the two targets themselves

Neither target is ever averaged with the other or used as a substitute for it.

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{status, model_loaded, checkpoint_sha256, n_samples}` |
| GET | `/model` | model card: architecture, config, normalisers, provenance, training config, measured test metrics (both targets), dataset attribution |
| GET | `/samples` | the 24 bundled held-out **test** samples: id, test index, fraction of high permeability, PDE residual of the official target |
| GET | `/samples/{id}?resolution=64\|128` | permeability field plus both targets |
| POST | `/predict` | prediction for a user field or a bundled sample |
| GET | `/activity?limit=1..100` | the most recent predictions served by this process. **Metadata only:** time, `input_kind`, sample id, resolution, strategy, latency, number of warnings, and rel-L2 vs official (benchmark examples only). In memory; resets on restart. Added in M5. |

Example `/predict` request:

```json
{ "permeability": [[...]],          // 64x64 or 128x128, a > 0, indexed [x][y] at cell centres (i+0.5)/n
  "sample_id": "test-2113",         // exactly one of permeability / sample_id
  "resolution": 128,                // sample_id only
  "strategy": null,                 // fno_native (64) | fno64_interpolated (128, default) | fno128_native (128)
  "return_targets": false }         // sample_id only
```

The response contains:
- `prediction`, `resolution`, `model_resolution`, `strategy`
- `inference_ms`: server-side encode, forward, decode and interpolation
- `checkpoint_sha256`, `warnings`, `input_summary`
- `input_kind`: `benchmark_example` or `custom`
- for samples: `evaluation` and, if requested, `targets`

**Default for 128×128 is `fno64_interpolated`.** On the test split its error vs the official targets is 0.0320, against 0.0421 for zero-shot 128×128 inference (docs/RESULTS.md §2). `fno128_native` remains available. The trade-off: interpolation is worse on the last row and column (RESULTS.md §3).

## 4. Validation and errors

Error responses always have the shape `{"detail": ..., "request_id": ...}`.

| Input | Result |
|---|---|
| Missing both inputs, both given, unknown fields, `return_targets` with a user field, bad `strategy` / `resolution` | 422 (schema) |
| Grid not square, or size not 64 or 128 | 422. These are the only two resolutions with measured error; others are rejected rather than served unvalidated. |
| NaN / ±Inf, a ≤ 0, a > 1e6 | 422 |
| `fno128_native` on 64×64 input, or `fno_native` on 128×128 | 422 |
| Unknown `sample_id` | 404 |
| Body larger than `PDEFLOW_MAX_BODY_BYTES` (default 1 MB; a 128×128 JSON field is about 165 kB) | 413 |
| Values outside [0.1, 1.0], or non-binary values | **200 with `warnings`**. The training data are binary {0.1, 1.0}, so accuracy there is unknown. |

## 5. Logging

The service writes JSON lines to stderr, and to `PDEFLOW_LOG_FILE` if set. Events:
- `startup`: checkpoint path, SHA, load time, thread count
- `request`: request_id, method, path, status, duration_ms
- `prediction`: request_id, sample_id, resolution, strategy, inference_ms, number of warnings, rel-L2 vs official for samples
- `unhandled_error`: with traceback; the client receives a generic 500

`X-Request-ID` is accepted if it matches `[A-Za-z0-9._-]{1,64}`, otherwise a fresh one is generated. It is echoed in the response.

**Concurrency:** one model forward runs at a time (a lock), using `PDEFLOW_NUM_THREADS` intra-op threads. This avoids the CPU oversubscription measured during M2.

## 6. Measured HTTP latency

Source: `reports/m4_api_latency.json`. Setup: uvicorn with one worker, 4 vCPU, client on the same host, sequential requests, 30 repetitions after 3 warm-ups.

| Request | Response size | Median wall ms | p95 wall ms | Server `inference_ms` median |
|---|---|---|---|---|
| sample, 64×64, fno_native | 83 kB | 38.0 | 63.5 | 32.7 |
| sample, 128×128, fno64_interpolated | 330 kB | 40.1 | 53.5 | 33.0 |
| sample, 128×128, fno128_native | 330 kB | 55.9 | 87.3 | 49.0 |
| user field 128×128 (165 kB request), default | 329 kB | 40.0 | 55.9 | 27.1 |

Server inference here is slower than the forward-only benchmark in RESULTS.md §5. It includes normalisation, decoding and interpolation, and runs in FastAPI's worker thread. Both figures are reported as measured.

## 7. Configuration (environment variables, no secrets)

These are listed in `pdeflow/api/settings.py`:
- `PDEFLOW_CHECKPOINT`, `PDEFLOW_EXPECTED_CHECKPOINT_SHA256`
- `PDEFLOW_SAMPLES_FILE`, `PDEFLOW_EVAL_REPORT`
- `PDEFLOW_NUM_THREADS`, `PDEFLOW_CORS_ORIGINS` (default `http://localhost:3000`)
- `PDEFLOW_MAX_BODY_BYTES`, `PDEFLOW_LOG_FILE`

## 8. Bundled samples

`backend/assets/demo_samples_v1.npz` (2.8 MB) contains 24 held-out **test** samples, built by `scripts/build_demo_samples.py`:
- test idx 2113 and 4701 (the RESULTS figure examples)
- 22 more drawn with `default_rng(0)`

PDEBench data are © the PDEBench authors, DaRUS doi:10.18419/darus-2986, licensed **CC BY 4.0** (licence read from the DaRUS API). Attribution is stored in the file and shown by `/model`. The test `test_bundle_matches_processed_test_split` checks that the samples equal the processed test split.
