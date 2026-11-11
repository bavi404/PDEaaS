import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("plotly.js-dist-min", () => ({ default: { react: vi.fn(() => Promise.resolve()), purge: vi.fn() } }));

import { ResultsView } from "@/components/ResultsView";
import type { PredictionRecord } from "@/lib/store";
import { benchmarkResponse, customResponse, grid } from "./fixtures";

const record = (r: PredictionRecord["response"], source: PredictionRecord["source"]): PredictionRecord => ({
  createdAt: "2026-11-11T12:00:00.000Z",
  source,
  label: "label",
  permeability: grid(r.resolution, () => 0.1),
  request: {},
  response: r,
});

describe("ResultsView", () => {
  it("benchmark example: comparison layout with the official target and a separate secondary section", () => {
    render(<ResultsView record={record(benchmarkResponse(64), "benchmark_example")} />);
    for (const id of ["hm-input", "hm-prediction", "hm-official", "hm-error"]) expect(screen.getByTestId(id)).toBeInTheDocument();
    expect(screen.getByTestId("stat-primary")).toHaveTextContent("Relative L2 vs official PDEBench target");
    expect(screen.getByTestId("stat-primary")).toHaveTextContent("1.34%");
    expect(screen.getByTestId("stat-latency")).toHaveTextContent("33.2 ms");
    const sec = screen.getByTestId("secondary-analysis");
    expect(within(sec).getByTestId("hm-reference")).toBeInTheDocument();
    expect(within(sec).getByTestId("stat-secondary")).toHaveTextContent("1.31%");
    expect(within(sec).getByTestId("stat-gap")).toHaveTextContent("0.06%");
    // the reference never appears in the primary comparison
    expect(within(screen.getByText("Comparison with the official PDEBench target").closest("section")!).queryByTestId("hm-reference")).toBeNull();
    expect(screen.queryByTestId("no-ground-truth")).toBeNull();
    expect(screen.getByText(/Benchmark example/)).toBeInTheDocument();
  });

  it("custom input: no ground truth, no error map, no error metric", () => {
    render(<ResultsView record={record(customResponse(128), "custom")} />);
    expect(screen.getByTestId("hm-input")).toBeInTheDocument();
    expect(screen.getByTestId("hm-prediction")).toBeInTheDocument();
    for (const id of ["hm-official", "hm-error", "hm-reference", "stat-primary", "secondary-analysis"]) expect(screen.queryByTestId(id)).toBeNull();
    expect(screen.getByTestId("no-ground-truth")).toHaveTextContent("no ground truth");
    expect(screen.queryByText(/Relative L2/)).toBeNull();
    expect(screen.getByText(/Custom input · no ground truth/)).toBeInTheDocument();
  });

  it("shows the API's out-of-distribution warnings", () => {
    const w = ["non-binary field: training data contain only the values 0.1 and 1.0; accuracy unknown"];
    render(<ResultsView record={record(customResponse(64, w), "custom")} />);
    expect(screen.getByTestId("ood-warning")).toHaveTextContent(w[0]);
  });

  it("does not show metrics if a benchmark record lacks an evaluation", () => {
    const r = { ...benchmarkResponse(64), evaluation: undefined, targets: undefined };
    render(<ResultsView record={record(r, "benchmark_example")} />);
    expect(screen.queryByTestId("stat-primary")).toBeNull();
    expect(screen.queryByTestId("hm-official")).toBeNull();
  });
});
