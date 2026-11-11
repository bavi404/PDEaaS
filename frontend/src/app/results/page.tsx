"use client";

import Link from "next/link";
import { ResultsView } from "@/components/ResultsView";
import { EmptyState, LoadingState, PageHeader } from "@/components/ui";
import { api } from "@/lib/api";
import { usePredictionStore } from "@/lib/store";
import { useResource } from "@/lib/useResource";

export default function ResultsPage() {
  const { latest, hydrated } = usePredictionStore();
  const model = useResource(() => api.model());
  return (
    <>
      <PageHeader
        eyebrow="Results"
        title="Prediction results"
        actions={
          <Link className="btn btn--secondary" href="/workspace">
            New prediction
          </Link>
        }
      >
        Pressure field predicted by the FNO. For benchmark examples, the result is compared with the official PDEBench
        target; the independent steady-state reference is shown separately as secondary analysis.
      </PageHeader>
      {!hydrated && <LoadingState />}
      {hydrated && !latest && (
        <EmptyState
          title="No prediction yet"
          action={
            <Link className="btn btn--primary" href="/workspace">
              Open the solver workspace
            </Link>
          }
        >
          Run a prediction in the solver workspace to see it here.
        </EmptyState>
      )}
      {hydrated && latest && <ResultsView record={latest} serviceVersion={model.status === "success" ? model.data.service_version : undefined} />}
    </>
  );
}
