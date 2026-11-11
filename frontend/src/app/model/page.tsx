"use client";

import { MetricsTable } from "@/components/MetricsTable";
import { Card, ErrorState, KeyValue, LoadingState, Notice, PageHeader } from "@/components/ui";
import { api, type ModelCard } from "@/lib/api";
import { int, pct } from "@/lib/format";
import { fmt, limitations } from "@/lib/limitations";
import { useResource } from "@/lib/useResource";

export default function ModelPage() {
  const model = useResource(() => api.model());
  return (
    <>
      <PageHeader eyebrow="Model" title="Model information">
        Architecture, training setup and held-out evaluation of the served checkpoint, as reported by the API.
      </PageHeader>
      {model.status === "loading" && <LoadingState label="Loading model card…" />}
      {model.status === "error" && <ErrorState error={model.error} onRetry={model.reload} what="the model card" />}
      {model.status === "success" && <ModelDetails card={model.data} />}
    </>
  );
}

function ModelDetails({ card }: { card: ModelCard }) {
  const ck = card.checkpoint;
  const cfg = ck.model_config;
  const train = ck.training_config.train ?? {};
  const data = ck.training_config.data ?? {};
  const prov = ck.provenance as Record<string, string | number | boolean | undefined>;
  const tm = card.test_metrics;
  return (
    <div className="stack" data-testid="model-details">
      <div className="grid grid--2">
        <Card title="Architecture" subtitle={ck.architecture}>
          <KeyValue
            rows={[
              ["Operator", "Fourier Neural Operator (neuraloperator 2.0.0)"],
              ["Fourier modes", <span key="m" className="mono" data-testid="model-modes">{cfg.n_modes.join(" × ")}</span>],
              ["Hidden channels", <span key="h" className="mono" data-testid="model-hidden">{cfg.hidden_channels}</span>],
              ["Fourier layers", <span key="l" className="mono" data-testid="model-layers">{cfg.n_layers}</span>],
              ["Input / output channels", <span key="io" className="mono">{cfg.in_channels} → {cfg.out_channels}</span>],
              ["Parameters", <span key="p" className="mono" data-testid="model-params">{int(ck.n_params)}</span>],
              ["Positional encoding", "grid coordinates appended to the input (library default)"],
            ]}
          />
          <div className="divider" />
          <p className="small muted">
            Each Fourier layer applies a learned linear map to the lowest {cfg.n_modes[0]}×{cfg.n_modes[1]} Fourier modes of the
            hidden state and adds a pointwise linear path, followed by a nonlinearity. Because the spectral weights act on
            modes rather than grid points, the same weights can be evaluated on 64×64 and 128×128 grids.
          </p>
        </Card>

        <Card title="Training" subtitle="Official PDEBench targets only">
          <KeyValue
            rows={[
              ["Dataset", card.dataset.split(". Subset")[0]],
              ["Training / validation samples", <span key="n" className="mono">{fmt(data.n_train)} / {fmt(data.n_val)}</span>],
              ["Training resolution", <span key="r" className="mono">{ck.training_resolution}×{ck.training_resolution} (stride {fmt(data.stride)} from 128×128)</span>],
              ["Loss", <span key="lo" className="mono">{fmt(train.loss)} (on physical u)</span>],
              ["Optimiser", <span key="o" className="mono">AdamW · lr {fmt(train.lr)} · wd {fmt(train.weight_decay)} · batch {fmt(train.batch_size)}</span>],
              ["Schedule", <span key="s" className="mono">{fmt(train.scheduler)} · {fmt(train.epochs)} epochs</span>],
              ["Model selection", <span key="ms" className="mono">best {ck.selection_metric.name} = {pct(ck.selection_metric.value)} at epoch {ck.epoch}</span>],
              ["Normalisation", <span key="no" className="mono">a: μ {ck.normalizers.a.mean.toFixed(4)}, σ {ck.normalizers.a.std.toFixed(4)} · u: μ {ck.normalizers.u.mean.toFixed(4)}, σ {ck.normalizers.u.std.toFixed(4)}</span>],
            ]}
          />
        </Card>
      </div>

      <Card title="Evaluation" subtitle={tm ? `${tm.metric}, ${tm.n_test} held-out test samples (${tm.source})` : undefined} testId="evaluation-card">
        {tm ? (
          <div className="stack">
            <MetricsTable metrics={tm} />
            <div className="grid grid--2">
              <Notice tone="info" title="Primary: official PDEBench target">
                {card.targets.official_pdebench_target}.
              </Notice>
              <div className="notice" style={{ background: "var(--secondary-tint)", borderColor: "#e3dff0", color: "var(--secondary-ink)" }}>
                <strong className="notice__title">Secondary: independent steady-state reference</strong>
                {card.targets.independent_steady_state_reference}.
              </div>
            </div>
          </div>
        ) : (
          <Notice tone="warning">No evaluation report matches the loaded checkpoint, so no evaluation results are shown.</Notice>
        )}
      </Card>

      <div className="grid grid--2">
        <Card title="Limitations" testId="limitations">
          <ul className="prose">
            {limitations(card).map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Card>
        <Card title="Checkpoint and provenance">
          <KeyValue
            rows={[
              ["SHA-256", <span key="sha" className="mono break" data-testid="model-sha">{ck.sha256}</span>],
              ["File / format", <span key="f" className="mono">{ck.file} · {ck.format}</span>],
              ["Service version", <span key="v" className="mono">{card.service_version}</span>],
              ["Supported resolutions", <span key="sr" className="mono">{ck.supported_resolutions.join(", ")}</span>],
              ["Framework", <span key="fw" className="mono">torch {fmt(prov.torch)} · Python {fmt(prov.python)}</span>],
              ["Training data SHA-256", <span key="td" className="mono break">{fmt(prov.train_file_sha256)}</span>],
              ["Trained", <span key="t" className="mono">{fmt(prov.started_utc)}</span>],
            ]}
          />
        </Card>
      </div>

      <Card title="Licences and attribution" testId="licensing">
        <KeyValue
          rows={[
            ["PDEFlow code and checkpoint", "MIT licence"],
            [
              "Bundled benchmark examples",
              <span key="d">
                PDEBench data (Takamoto et al., NeurIPS 2022), DaRUS{" "}
                <a href="https://doi.org/10.18419/darus-2986">doi:10.18419/darus-2986</a>, licensed{" "}
                <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. 24 held-out test samples; the
                steady-state reference fields are this project&apos;s adaptation.
              </span>,
            ],
            [
              "Third-party software",
              <span key="t">
                Each package keeps its own licence: <a href="/third-party-licenses.txt">third-party licence texts</a>
              </span>,
            ],
          ]}
        />
      </Card>
    </div>
  );
}
