# Hosting status and limitations

**Status (2026-11-12): no public deployment is live.** The application is complete and verified locally: the full test
suites, a Docker container under Hugging Face-like and Render-like limits, and a staged Hugging Face Space run. The
evidence is in `reports/` and the documents linked below. This page records what was tried, why it failed, and the
constraints of the remaining zero-cost option.

## 1. Why there is no public URL yet

| Option | Cost | Outcome |
|---|---|---|
| Hugging Face **Docker** Space (M6 plan) | needs a **paid PRO plan** to create (current HF docs) | not attempted; correction in [`DEPLOYMENT.md`](DEPLOYMENT.md) |
| Hugging Face **ZeroGPU Gradio** Space, CPU-only inference ([`ZEROGPU.md`](ZEROGPU.md)) | free for eligible personal accounts | **two controlled attempts, both failed** (below) |
| **Render Free** web service with the unchanged Docker image | free (no payment method required per Render docs) | configuration prepared and emulated locally; **not created** (requires the owner's Render account) |

**ZeroGPU attempt 1** (2026-11-11 19:14 UTC) failed at **build**. HF's build installs our requirements together with
`gradio[oauth,mcp]`, whose `mcp` extra needs `pydantic<=2.12.5`, while the lock pinned 2.14.3. Fixed in `0172fb2`, and
`deploy/zerogpu/check_hf_build.py` now reproduces HF's exact install command with pip before any deployment.

**ZeroGPU attempt 2** (19:53 UTC) **built successfully**. The app started inside the Space: it verified the checkpoint
SHA-256 and loaded the model and samples. It was then stopped by the platform for two reasons:
1. `No @spaces.GPU function detected during startup`: **HF's ZeroGPU runtime requires GPU-decorated functions**,
   so a CPU-only app is not accepted there.
2. Port 7860 was already in use when uvicorn tried to bind it, because the app is served by uvicorn rather than
   Gradio's own `launch()`.

Making ZeroGPU work would mean running inference on the GPU inside `@spaces.GPU`. That uses visitors' daily GPU quota
(2 min/day unauthenticated), adds queueing, and changes the numerics (cuFFT vs CPU FFT), which **cannot be validated
in this GPU-less environment**. Adding an unused GPU function just to pass the check would circumvent the platform's
intent, so it was not done.

## 2. Limitations of the zero-cost hosting that remains (Render Free)

Measured with the unchanged image under `--memory=512m --memory-swap=512m`, fractional CPU and `PORT=10000`:

- **Cold starts.**
  - A Free web service **spins down after 15 minutes without traffic**; Render's own wake-up "takes about one minute".
  - The app then needs **24 s (0.5 CPU) to 172 s (0.1 CPU)** to import PyTorch, verify the checkpoint and load the model.
  - The first visit after idling may therefore wait about **1.5–4 minutes**; later requests are fast.
- **Free-tier constraints.**
  - **512 MB RAM** and **"less than 1 CPU"**; the exact CPU share is not documented.
  - 750 free instance-hours per month; an ephemeral filesystem; no persistent disk.
  - Outbound bandwidth and build minutes are capped monthly. Without a payment method, Render **suspends** the service instead of billing.
  - Measured memory: about **290 MiB** of non-reclaimable memory; per-process peaks of 316 MiB (API) and 98 MiB (web); no OOM.
- **CPU inference.**
  - There is no GPU; the FNO runs on CPU by design. This keeps results **bit-identical** to the evaluated benchmark (verified on all 1,000 test samples).
  - Warm prediction latency: **0.08–0.12 s at 0.5 CPU** and **0.4–0.7 s at 0.1 CPU** for a 128×128 field, against 12–16 ms server-side inference on 2 dedicated vCPUs.
  - One model forward runs at a time (an API lock), so concurrent visitors queue. The health endpoint stayed responsive (< 0.6 s) during 4 concurrent predictions.
- **Activity feed:** the dashboard's "recent prediction activity" is kept in memory, so it **resets** on every spin-down or redeploy.
- **Platform terms can change.** All free-tier statements above reflect the providers' documentation as read on 2026-11-11.

## 3. Running it yourself

The same image runs anywhere Docker runs (`docker build -t pdeflow . && docker run -p 7860:7860 pdeflow`; see
[`DOCKER.md`](DOCKER.md)). It needs about 0.3 GB of RAM and no GPU.
