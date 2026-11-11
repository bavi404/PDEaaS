"use client";

import type { Activity } from "@/lib/api";
import { ms, pct, relativeTime } from "@/lib/format";
import { Badge, EmptyState } from "./ui";
import Link from "next/link";

export function ActivityList({ activity, now }: { activity: Activity; now?: Date }) {
  if (activity.items.length === 0)
    return (
      <EmptyState title="No predictions yet" action={<Link className="btn btn--secondary" href="/workspace">Open the solver workspace</Link>}>
        Predictions served by this API process will appear here.
      </EmptyState>
    );
  return (
    <div className="table-wrap">
      <table className="table table--activity" data-testid="activity-table">
        <thead>
          <tr>
            <th scope="col">When</th>
            <th scope="col">Input</th>
            <th scope="col">Grid</th>
            <th scope="col" className="num">Latency</th>
            <th scope="col" className="num">Rel. L2 vs official</th>
          </tr>
        </thead>
        <tbody>
          {activity.items.map((it, i) => (
            <tr key={`${it.timestamp_utc}-${i}`}>
              <td className="muted nowrap">{relativeTime(it.timestamp_utc, now)}</td>
              <td>
                {it.input_kind === "benchmark_example" ? <Badge tone="accent">Benchmark</Badge> : <Badge>Custom</Badge>}{" "}
                <span className="mono small">{it.sample_id ?? "user field"}</span>
                {it.n_warnings > 0 && (
                  <>
                    {" "}
                    <Badge tone="warn">OOD</Badge>
                  </>
                )}
              </td>
              <td className="mono small nowrap">
                {it.resolution}²{it.strategy === "fno64_interpolated" ? " (interp.)" : it.strategy === "fno128_native" ? " (direct)" : ""}
              </td>
              <td className="num mono">{ms(it.inference_ms)}</td>
              <td className="num mono">{it.rel_l2_vs_official === null ? <span className="muted">n/a</span> : pct(it.rel_l2_vs_official)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
