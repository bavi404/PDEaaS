"use client";

import Link from "next/link";
import { useEffect } from "react";
import { ActivityList } from "@/components/ActivityList";
import { MetricsTable } from "@/components/MetricsTable";
import { Card, ErrorState, KeyValue, LoadingState, Notice, PageHeader, Stat } from "@/components/ui";
import { PRIMARY, api } from "@/lib/api";
import { int, pct, shortSha } from "@/lib/format";
import { useResource } from "@/lib/useResource";

const ACTIVITY_POLL_MS = 15_000;

export default function DashboardPage() {
  const model = useResource(() => api.model());
  const activity = useResource(() => api.activity(8));
  const { reload: reloadActivity } = activity;

  useEffect(() => {
    const t = setInterval(reloadActivity, ACTIVITY_POLL_MS);
    return () => clearInterval(t);
  }, [reloadActivity]);

  const ck = model.status === "success" ? model.data.checkpoint : null;
  const tm = model.status === "success" ? model.data.test_metrics : null;

  return (
    <>
      <PageHeader
        eyebrow="Dashboard"
        title="Neural-operator surrogate for 2D Darcy flow"
        actions={
          <Link className="btn btn--primary" href="/workspace">
            Open solver workspace
          </Link>
        }
      >
        PDEFlow serves a Fourier Neural Operator (FNO) trained on the PDEBench 2D Darcy Flow benchmark. Given a
        permeability field, it returns the steady pressure field in tens of milliseconds on the CPUs we tested. Every number on this page is
        read from the running API.
      </PageHeader>

      <div className="stack">
        {model.status === "loading" && <LoadingState label="Loading model card…" />}
        {model.status === "error" && <ErrorState error={model.error} onRetry={model.reload} what="the model card" />}
        {ck && (
          <div className="stat-row" data-testid="dashboard-stats">
            <Stat label="Model" value="FNO 2D" hint={`${int(ck.n_params)} parameters`} />
            <Stat label="Fourier modes" value={ck.model_config.n_modes.join(" × ")} hint={`${ck.model_config.n_layers} layers · width ${ck.model_config.hidden_channels}`} />
            <Stat
              testId="stat-test-64"
              label="Test rel. L2 · 64×64"
              value={tm ? pct(tm.res64_fno_native[PRIMARY].mean) : "n/a"}
              hint={tm ? `mean over ${tm.n_test} held-out samples, vs official target` : "metrics unavailable for this checkpoint"}
            />
            <Stat
              testId="stat-test-128"
              label="Test rel. L2 · 128×128"
              value={tm ? pct(tm.res128_fno64_interpolated[PRIMARY].mean) : "n/a"}
              hint="default 128×128 path (64×64 + interpolation)"
            />
            <Stat label="Checkpoint" value={<span className="mono">{shortSha(ck.sha256)}</span>} hint={`epoch ${ck.epoch} · service v${model.status === "success" ? model.data.service_version : ""}`} />
          </div>
        )}

        <div className="grid grid--2">
          <Card title="PDE: steady 2D Darcy flow" subtitle="PDEBench, forcing β = 1.0">
            <div className="equation" aria-label="minus divergence of a times gradient of u equals beta">
              −∇·( a(x) ∇u(x) ) = β,&nbsp;&nbsp; x ∈ (0, 1)²,&nbsp;&nbsp; u = 0 on ∂(0, 1)²
            </div>
            <div className="divider" />
            <KeyValue
              rows={[
                ["Input", "permeability a(x, y) > 0 (binary 0.1 / 1.0 in the benchmark)"],
                ["Output", "pressure u(x, y)"],
                ["Grid", "cell-centred, 128×128 native; model trained at 64×64"],
                ["Benchmark target", "official PDEBench solution (primary)"],
                ["Secondary check", "independent steady-state reference, reported separately"],
              ]}
            />
          </Card>

          <Card title="Model performance" subtitle={tm ? `Relative L2 on ${tm.n_test} held-out test samples` : undefined} testId="performance-card">
            {model.status === "loading" && <LoadingState />}
            {model.status === "error" && <ErrorState error={model.error} onRetry={model.reload} />}
            {model.status === "success" && !tm && (
              <Notice tone="warning">No evaluation report matches the loaded checkpoint, so no test metrics are shown.</Notice>
            )}
            {tm && (
              <div className="stack">
                <MetricsTable metrics={tm} compact />
                <p className="small muted">
                  The model is trained and selected on the <strong>official PDEBench targets</strong>; the shaded column
                  compares against an independent exact steady-state solve for analysis only.{" "}
                  <Link href="/model">Full evaluation →</Link>
                </p>
              </div>
            )}
          </Card>
        </div>

        <Card
          title="Recent prediction activity"
          subtitle={activity.status === "success" ? `${activity.data.total_since_start} predictions since the API started · refreshes every 15 s` : undefined}
          actions={
            <button type="button" className="btn btn--secondary" onClick={reloadActivity}>
              Refresh
            </button>
          }
          testId="activity-card"
        >
          {activity.status === "loading" && <LoadingState label="Loading activity…" />}
          {activity.status === "error" && <ErrorState error={activity.error} onRetry={reloadActivity} what="activity" />}
          {activity.status === "success" && <ActivityList activity={activity.data} />}
        </Card>
      </div>
    </>
  );
}
