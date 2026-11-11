"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Heatmap } from "@/components/Heatmap";
import { Badge, Card, EmptyState, ErrorState, LoadingState, Notice, PageHeader, Segmented } from "@/components/ui";
import { CustomFieldEditor, DEFAULT_GENERATOR, type CustomSource } from "@/components/workspace/CustomFieldEditor";
import { ExamplePicker } from "@/components/workspace/ExamplePicker";
import { ApiError, api, type Grid, type PredictRequest, type Resolution, type Strategy } from "@/lib/api";
import { generateField, validateGrid, type FieldCheck, type GeneratorParams } from "@/lib/fields";
import { STRATEGY_LABEL, pct, sci } from "@/lib/format";
import { usePredictionStore } from "@/lib/store";
import { useResource } from "@/lib/useResource";

type Mode = "benchmark" | "custom";

export default function WorkspacePage() {
  const router = useRouter();
  const { setLatest } = usePredictionStore();
  const [mode, setMode] = useState<Mode>("benchmark");
  const [resolution, setResolution] = useState<Resolution>(128);
  const [strategy128, setStrategy128] = useState<Strategy>("fno64_interpolated");
  const [selected, setSelected] = useState<string | null>(null);
  const [source, setSource] = useState<CustomSource>("generator");
  const [gen, setGen] = useState<GeneratorParams>(DEFAULT_GENERATOR);
  const [pasted, setPasted] = useState<FieldCheck | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<ApiError | null>(null);

  const samples = useResource(() => api.samples());
  useEffect(() => {
    if (samples.status === "success" && !selected && samples.data.length) setSelected(samples.data[0].id);
  }, [samples, selected]);
  const sample = useResource(() => (selected ? api.sample(selected, resolution) : Promise.reject(new ApiError("not_found", "No example selected"))), [selected, resolution]);

  // ---- custom input
  const generated = useMemo(() => (mode === "custom" && source === "generator" ? generateField({ ...gen, resolution }) : null), [mode, source, gen, resolution]);
  const customCheck: FieldCheck | null = useMemo(() => {
    if (mode !== "custom") return null;
    if (source === "generator") return generated ? validateGrid(generated) : null;
    return pasted;
  }, [mode, source, generated, pasted]);
  const effectiveResolution: Resolution = mode === "custom" && source === "paste" && pasted?.ok ? pasted.resolution : resolution;
  const strategy: Strategy = effectiveResolution === 64 ? "fno_native" : strategy128;

  // ---- what will be sent
  let inputGrid: Grid | null = null;
  let blocked: string | null = null;
  if (mode === "benchmark") {
    if (sample.status === "success") inputGrid = sample.data.permeability;
    else blocked = "Select a benchmark example.";
  } else if (!customCheck) blocked = source === "paste" ? "Paste or upload a field." : "Configure the generator.";
  else if (!customCheck.ok) blocked = customCheck.error;
  else inputGrid = customCheck.grid;

  async function runPrediction() {
    if (!inputGrid) return;
    setSubmitting(true);
    setSubmitError(null);
    const request: PredictRequest =
      mode === "benchmark"
        ? { sample_id: selected!, resolution: effectiveResolution, strategy, return_targets: true }
        : { permeability: inputGrid, strategy };
    try {
      const response = await api.predict(request);
      setLatest({
        createdAt: new Date().toISOString(),
        source: response.input_kind,
        label: mode === "benchmark" ? `${selected} · ${effectiveResolution}×${effectiveResolution}` : source === "generator" ? `generated · seed ${gen.seed} · ${effectiveResolution}×${effectiveResolution}` : `uploaded field · ${effectiveResolution}×${effectiveResolution}`,
        permeability: inputGrid,
        request: mode === "benchmark" ? request : { ...request, permeability: undefined },
        response,
      });
      router.push("/results");
    } catch (e) {
      setSubmitError(e instanceof ApiError ? e : new ApiError("server", String(e)));
      setSubmitting(false);
    }
  }

  const sampleMeta = mode === "benchmark" && samples.status === "success" ? samples.data.find((s) => s.id === selected) : undefined;

  return (
    <>
      <PageHeader eyebrow="Solver workspace" title="Set up a prediction">
        Choose a held-out benchmark example, for which the official PDEBench solution is available for comparison, or
        provide your own permeability field.
      </PageHeader>

      <div className="grid grid--sidebar">
        <div className="stack">
          <Card title="1 · PDE">
            <button type="button" className="choice is-active" aria-pressed="true" data-testid="pde-darcy">
              <span>
                <strong>2D Darcy flow (steady)</strong>
                <span className="block small muted">−∇·(a∇u) = 1 on (0,1)², u = 0 on the boundary · PDEBench β = 1.0</span>
              </span>
            </button>
            <p className="small muted" style={{ marginTop: 8 }}>
              This deployment serves one trained operator; other PDE families are not offered.
            </p>
          </Card>

          <Card title="2 · Input">
            <Segmented<Mode>
              label="Input type"
              name="mode"
              value={mode}
              onChange={(m) => {
                setMode(m);
                setSubmitError(null);
              }}
              options={[
                { value: "benchmark", label: "Benchmark example" },
                { value: "custom", label: "Custom field" },
              ]}
            />
            <div className="divider" />
            {mode === "benchmark" ? (
              <>
                {samples.status === "loading" && <LoadingState label="Loading examples…" />}
                {samples.status === "error" && <ErrorState error={samples.error} onRetry={samples.reload} what="examples" />}
                {samples.status === "success" &&
                  (samples.data.length ? (
                    <>
                      <p className="small muted" style={{ marginBottom: 8 }}>
                        {samples.data.length} samples from the held-out test split (never used in training or model selection).
                      </p>
                      <ExamplePicker samples={samples.data} selected={selected} onSelect={setSelected} />
                    </>
                  ) : (
                    <EmptyState title="No examples bundled with this API" />
                  ))}
              </>
            ) : (
              <CustomFieldEditor source={source} onSourceChange={setSource} params={gen} onParamsChange={setGen} pasted={pasted} onPasted={setPasted} />
            )}
          </Card>

          <Card title="3 · Resolution and inference">
            <div className="stack" style={{ gap: 14 }}>
              <Segmented<Resolution>
                label={mode === "custom" && source === "paste" ? "Resolution (from the uploaded field)" : "Grid resolution"}
                name="resolution"
                value={effectiveResolution}
                onChange={setResolution}
                options={[
                  { value: 64, label: "64 × 64", disabled: mode === "custom" && source === "paste" },
                  { value: 128, label: "128 × 128", disabled: mode === "custom" && source === "paste" },
                ]}
              />
              {effectiveResolution === 128 ? (
                <Segmented<Strategy>
                  label="128×128 inference path"
                  name="strategy"
                  value={strategy128}
                  onChange={setStrategy128}
                  options={[
                    { value: "fno64_interpolated", label: "64×64 + interpolation (recommended)" },
                    { value: "fno128_native", label: "Direct 128×128 (zero-shot)" },
                  ]}
                />
              ) : (
                <p className="small muted">At 64×64 the model runs at its training resolution.</p>
              )}
              <p className="small muted" data-testid="strategy-label">
                {STRATEGY_LABEL[strategy]}
              </p>
              {submitError && (
                <Notice tone="error" title={submitError.kind === "unavailable" ? "API unavailable" : "Prediction failed"} testId="submit-error">
                  {submitError.message}
                  {submitError.requestId && <span className="block mono small">request id {submitError.requestId}</span>}
                </Notice>
              )}
              <button type="button" className="btn btn--primary btn--block" disabled={!inputGrid || submitting} onClick={runPrediction} data-testid="predict-button">
                {submitting ? "Running FNO…" : "Run prediction"}
              </button>
              {blocked && !submitting && (
                <p className="small muted" data-testid="blocked-reason">
                  {blocked}
                </p>
              )}
            </div>
          </Card>
        </div>

        <div className="stack">
          <Card
            title="Input field"
            subtitle={mode === "benchmark" ? "Benchmark example: the official target is held back until you run the prediction" : "Custom input: no ground truth is available"}
            actions={mode === "benchmark" ? <Badge tone="accent">Benchmark</Badge> : <Badge>Custom</Badge>}
            testId="input-preview"
          >
            {mode === "benchmark" && sample.status === "loading" && selected && <LoadingState label="Loading example…" />}
            {mode === "benchmark" && sample.status === "error" && selected && <ErrorState error={sample.error} onRetry={sample.reload} what="the example" />}
            {mode === "custom" && customCheck && !customCheck.ok && (
              <Notice tone="error" title="Invalid field" testId="field-error">
                {customCheck.error}
              </Notice>
            )}
            {mode === "custom" && !customCheck && <EmptyState title="No field yet">Paste data or upload a file to preview it here.</EmptyState>}
            {inputGrid && (
              <div className="stack" style={{ gap: 12 }}>
                <Heatmap testId="hm-workspace-input" title="Permeability a(x, y)" valueLabel="a" colorscale="permeability" z={inputGrid} height={420} />
                {sampleMeta && (
                  <p className="small muted">
                    Test sample #{sampleMeta.test_index} · {pct(sampleMeta.fraction_high_permeability, 1)} of cells at a = 1.0 · PDE residual of the
                    official target {sci(sampleMeta.official_target_pde_residual)}
                  </p>
                )}
                {mode === "custom" && customCheck?.ok && customCheck.warnings.length > 0 && (
                  <Notice tone="warning" title="Outside the training distribution" testId="ood-preview">
                    <ul className="plain-list">
                      {customCheck.warnings.map((w) => (
                        <li key={w}>{w}</li>
                      ))}
                    </ul>
                  </Notice>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
