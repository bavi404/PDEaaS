import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import { ACTIVITY, MODEL_CARD, SAMPLES, benchmarkResponse, customResponse, grid, sampleDetail } from "./fixtures";

vi.mock("plotly.js-dist-min", () => ({ default: { react: vi.fn(() => Promise.resolve()), purge: vi.fn() } }));

const { push, mockApi } = vi.hoisted(() => ({
  push: vi.fn(),
  mockApi: { health: vi.fn(), model: vi.fn(), samples: vi.fn(), sample: vi.fn(), activity: vi.fn(), predict: vi.fn() },
}));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }), usePathname: () => "/" }));
vi.mock("@/lib/api", async (orig) => ({ ...(await orig<typeof import("@/lib/api")>()), api: mockApi }));

import DashboardPage from "@/app/page";
import ModelPage from "@/app/model/page";
import ResultsPage from "@/app/results/page";
import WorkspacePage from "@/app/workspace/page";
import { PredictionProvider } from "@/lib/store";

const wrap = (ui: React.ReactNode) => render(<PredictionProvider>{ui}</PredictionProvider>);
const down = () => Promise.reject(new ApiError("unavailable", "The PDEFlow API could not be reached. Check that the backend is running."));

beforeEach(() => {
  Object.values(mockApi).forEach((f) => f.mockReset());
  push.mockReset();
  window.sessionStorage.clear();
  mockApi.health.mockResolvedValue({ status: "ok", model_loaded: true, checkpoint_sha256: MODEL_CARD.checkpoint.sha256, n_samples: 2 });
  mockApi.model.mockResolvedValue(MODEL_CARD);
  mockApi.samples.mockResolvedValue(SAMPLES);
  mockApi.sample.mockImplementation((id: string, n: 64 | 128) => Promise.resolve(sampleDetail(id, n)));
  mockApi.activity.mockResolvedValue(ACTIVITY);
});

describe("Dashboard", () => {
  it("shows model metrics and recent activity from the API", async () => {
    wrap(<DashboardPage />);
    expect(await screen.findByTestId("stat-test-64")).toHaveTextContent("2.93%");
    expect(screen.getByTestId("stat-test-128")).toHaveTextContent("3.20%");
    expect(screen.getByText("357,217 parameters")).toBeInTheDocument();
    const act = await screen.findByTestId("activity-table");
    expect(within(act).getByText("test-2113")).toBeInTheDocument();
    expect(within(act).getByText("n/a")).toBeInTheDocument(); // custom: no metric
    expect(within(act).getByText("OOD")).toBeInTheDocument();
  });

  it("shows the empty state when there is no activity", async () => {
    mockApi.activity.mockResolvedValue({ ...ACTIVITY, items: [], total_since_start: 0 });
    wrap(<DashboardPage />);
    expect(await screen.findByText("No predictions yet")).toBeInTheDocument();
  });

  it("shows error states with retry when the backend is unavailable", async () => {
    mockApi.model.mockImplementation(down);
    mockApi.activity.mockImplementation(down);
    wrap(<DashboardPage />);
    const errs = await screen.findAllByTestId("error-state");
    expect(errs.length).toBeGreaterThanOrEqual(2);
    expect(errs[0]).toHaveTextContent("API unavailable");
    mockApi.model.mockResolvedValue(MODEL_CARD);
    fireEvent.click(within(errs[0]).getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("stat-test-64")).toBeInTheDocument();
  });

  it("states that metrics are unavailable when the report does not match the checkpoint", async () => {
    mockApi.model.mockResolvedValue({ ...MODEL_CARD, test_metrics: null });
    wrap(<DashboardPage />);
    expect(await screen.findByText(/No evaluation report matches/)).toBeInTheDocument();
    expect(screen.getByTestId("stat-test-64")).toHaveTextContent("n/a");
  });
});

describe("Solver workspace", () => {
  it("selects a benchmark example and submits it with return_targets", async () => {
    mockApi.predict.mockResolvedValue(benchmarkResponse(128));
    wrap(<WorkspacePage />);
    await screen.findByTestId("example-test-2113");
    await waitFor(() => expect(mockApi.sample).toHaveBeenCalledWith("test-2113", 128));
    await userEvent.click(screen.getByTestId("example-test-4701"));
    await waitFor(() => expect(mockApi.sample).toHaveBeenCalledWith("test-4701", 128));
    expect(screen.getByTestId("example-test-4701")).toHaveAttribute("aria-selected", "true");
    await screen.findByTestId("hm-workspace-input");
    await userEvent.click(screen.getByTestId("predict-button"));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/results"));
    expect(mockApi.predict).toHaveBeenCalledWith({ sample_id: "test-4701", resolution: 128, strategy: "fno64_interpolated", return_targets: true });
    const stored = JSON.parse(window.sessionStorage.getItem("pdeflow.latestPrediction.v1")!);
    expect(stored.source).toBe("benchmark_example");
  });

  it("switching to 64x64 reloads the example and uses the native strategy", async () => {
    mockApi.predict.mockResolvedValue(benchmarkResponse(64));
    wrap(<WorkspacePage />);
    await screen.findByTestId("hm-workspace-input");
    await userEvent.click(screen.getByLabelText("64 × 64"));
    await waitFor(() => expect(mockApi.sample).toHaveBeenCalledWith("test-2113", 64));
    await waitFor(() => expect(screen.getByTestId("predict-button")).toBeEnabled());
    await userEvent.click(screen.getByTestId("predict-button"));
    await waitFor(() => expect(mockApi.predict).toHaveBeenCalledWith(expect.objectContaining({ resolution: 64, strategy: "fno_native" })));
  });

  it("submits a generated custom field without sample id or targets", async () => {
    mockApi.predict.mockResolvedValue(customResponse(128));
    wrap(<WorkspacePage />);
    await userEvent.click(screen.getByLabelText("Custom field"));
    await screen.findByTestId("hm-workspace-input");
    await userEvent.click(screen.getByLabelText("Direct 128×128 (zero-shot)"));
    await userEvent.click(screen.getByTestId("predict-button"));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/results"));
    const body = mockApi.predict.mock.calls[0][0];
    expect(body.sample_id).toBeUndefined();
    expect(body.return_targets).toBeUndefined();
    expect(body.strategy).toBe("fno128_native");
    expect(body.permeability).toHaveLength(128);
    expect(new Set(body.permeability.flat())).toEqual(new Set([0.1, 1.0]));
  });

  it("warns before submission when generator values are outside the training distribution", async () => {
    wrap(<WorkspacePage />);
    await userEvent.click(screen.getByLabelText("Custom field"));
    const high = screen.getByTestId("gen-high");
    fireEvent.change(high, { target: { value: "3" } });
    expect(await screen.findByTestId("ood-preview")).toHaveTextContent("outside the training range");
  });

  it("rejects malformed pasted input and keeps the predict button disabled", async () => {
    wrap(<WorkspacePage />);
    await userEvent.click(screen.getByLabelText("Custom field"));
    await userEvent.click(screen.getByLabelText("Paste or upload"));
    fireEvent.change(screen.getByTestId("paste-input"), { target: { value: "[[1, 2], [3, 4]]" } });
    expect(await screen.findByTestId("paste-error")).toHaveTextContent("Size 2×2 is not supported");
    expect(screen.getByTestId("field-error")).toBeInTheDocument();
    expect(screen.getByTestId("predict-button")).toBeDisabled();
    fireEvent.change(screen.getByTestId("paste-input"), { target: { value: "not a grid" } });
    expect(await screen.findByTestId("paste-error")).toHaveTextContent("must be square");
    expect(mockApi.predict).not.toHaveBeenCalled();
  });

  it("accepts an uploaded CSV and derives the resolution from it", async () => {
    mockApi.predict.mockResolvedValue(customResponse(64));
    wrap(<WorkspacePage />);
    await userEvent.click(screen.getByLabelText("Custom field"));
    await userEvent.click(screen.getByLabelText("Paste or upload"));
    const csv = grid(64, (i, j) => ((i + j) % 2 ? 1 : 0.1)).map((r) => r.join(",")).join("\n");
    const file = new File([csv], "field.csv", { type: "text/csv" });
    await act(async () => {
      await userEvent.upload(screen.getByTestId("file-input"), file);
    });
    expect(await screen.findByTestId("paste-ok")).toHaveTextContent("64×64");
    expect(screen.getByLabelText("64 × 64")).toBeChecked();
    await userEvent.click(screen.getByTestId("predict-button"));
    await waitFor(() => expect(mockApi.predict).toHaveBeenCalledWith(expect.objectContaining({ strategy: "fno_native" })));
  });

  it("shows the server's validation message when the API rejects a prediction", async () => {
    mockApi.predict.mockRejectedValue(new ApiError("validation", "permeability must be strictly positive", 422, "rid-9"));
    wrap(<WorkspacePage />);
    await screen.findByTestId("hm-workspace-input");
    await userEvent.click(screen.getByTestId("predict-button"));
    const err = await screen.findByTestId("submit-error");
    expect(err).toHaveTextContent("Prediction failed");
    expect(err).toHaveTextContent("permeability must be strictly positive");
    expect(err).toHaveTextContent("rid-9");
    expect(push).not.toHaveBeenCalled();
  });

  it("shows an unavailable-backend state for examples", async () => {
    mockApi.samples.mockImplementation(down);
    wrap(<WorkspacePage />);
    expect(await screen.findByTestId("error-state")).toHaveTextContent("API unavailable");
    expect(screen.getByTestId("predict-button")).toBeDisabled();
  });
});

describe("Results page", () => {
  it("shows the empty state before any prediction", async () => {
    wrap(<ResultsPage />);
    expect(await screen.findByTestId("empty-state")).toHaveTextContent("No prediction yet");
  });

  it("renders the stored prediction", async () => {
    window.sessionStorage.setItem(
      "pdeflow.latestPrediction.v1",
      JSON.stringify({ createdAt: "2026-11-11T12:00:00Z", source: "benchmark_example", label: "x", permeability: grid(64, () => 0.1), request: {}, response: benchmarkResponse(64) }),
    );
    wrap(<ResultsPage />);
    expect(await screen.findByTestId("results-view")).toHaveAttribute("data-kind", "benchmark_example");
    expect(screen.getByTestId("hm-official")).toBeInTheDocument();
  });
});

describe("Model page", () => {
  it("shows architecture, training data, evaluation and limitations from the model card", async () => {
    wrap(<ModelPage />);
    expect(await screen.findByTestId("model-modes")).toHaveTextContent("12 × 12");
    expect(screen.getByTestId("model-hidden")).toHaveTextContent("32");
    expect(screen.getByTestId("model-layers")).toHaveTextContent("4");
    expect(screen.getByTestId("model-params")).toHaveTextContent("357,217");
    expect(screen.getByTestId("metrics-res128_fno128_native")).toHaveTextContent("4.21%");
    const lim = screen.getByTestId("limitations");
    expect(lim).toHaveTextContent("4.21% mean test error versus 3.20%");
    expect(lim).toHaveTextContent("2000 training samples for 150 epochs");
    expect(screen.queryByText(/Subset of this project/)).toBeNull();
  });

  it("shows an error state when the backend is down", async () => {
    mockApi.model.mockImplementation(down);
    wrap(<ModelPage />);
    expect(await screen.findByTestId("error-state")).toHaveTextContent("API unavailable");
  });
});
