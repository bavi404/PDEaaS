import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const react = vi.fn((..._args: unknown[]) => Promise.resolve());
const purge = vi.fn();
vi.mock("plotly.js-dist-min", () => ({ default: { react: (...a: unknown[]) => react(...a), purge: (...a: unknown[]) => purge(...a) } }));

import { Heatmap, buildFigure, cellCentres, transpose } from "@/components/Heatmap";

describe("Heatmap", () => {
  beforeEach(() => react.mockClear());

  it("transposes [x][y] grids to Plotly's z[y][x]", () => {
    expect(transpose([[1, 2, 3], [4, 5, 6]])).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it("uses physical cell-centre coordinates", () => {
    expect(cellCentres(4)).toEqual([0.125, 0.375, 0.625, 0.875]);
  });

  it("builds a heatmap with axes, colour bar, hover values and a locked aspect ratio", () => {
    const z = [[0, 1], [2, 3]];
    const { data, layout } = buildFigure({ z, title: "Predicted pressure", valueLabel: "û", colorscale: "pressure", zmin: 0, zmax: 3 });
    const t = data[0] as Record<string, unknown>;
    expect(t.type).toBe("heatmap");
    expect(t.z).toEqual([[0, 2], [1, 3]]);
    expect(t.x).toEqual([0.25, 0.75]);
    expect(t.colorscale).toBe("Viridis");
    expect(t.zmin).toBe(0);
    expect(t.zauto).toBe(false);
    expect((t.colorbar as { title: { text: string } }).title.text).toBe("û");
    expect(String(t.hovertemplate)).toContain("û = %{z");
    expect(String(t.hovertemplate)).toContain("x = %{x");
    const l = layout as { xaxis: { title: { text: string } }; yaxis: { scaleanchor: string; title: { text: string } } };
    expect(l.xaxis.title.text).toBe("x");
    expect(l.yaxis.title.text).toBe("y");
    expect(l.yaxis.scaleanchor).toBe("x");
  });

  it("renders a labelled figure and draws it with Plotly", async () => {
    render(<Heatmap z={[[0.1, 1], [1, 0.1]]} title="Permeability a(x, y)" valueLabel="a" colorscale="permeability" testId="hm" />);
    expect(screen.getByText("Permeability a(x, y)")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", expect.stringContaining("2×2 heatmap, a from 0.1000 to 1.000"));
    expect(screen.getByTestId("hm")).toHaveTextContent("min 0.1000 · max 1.000");
    await waitFor(() => expect(react).toHaveBeenCalledTimes(1));
    const [, data] = react.mock.calls[0] as [HTMLElement, { colorscale: string }[]];
    expect(data[0].colorscale).toBe("Cividis");
  });
});
