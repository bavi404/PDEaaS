"use client";

/**
 * Interactive Plotly heatmap of a field on the unit square.
 * - Input grids are indexed [x][y] (API convention); Plotly expects z[row = y][col = x], so we transpose.
 * - Axes are physical coordinates of the cell centres (i + 0.5) / n, aspect ratio locked to 1.
 * - Hover shows x, y and the value with units/labels; a colour bar with a title is always shown.
 * - Panels that must be compared (prediction vs target) share zmin/zmax, passed in by the parent.
 * plotly.js is loaded lazily on the client (it needs `window`).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { Grid } from "@/lib/api";

export type ColorScaleName = "pressure" | "permeability" | "error";

// Viridis (pressure), Cividis (permeability; colour-blind safe), Inferno anchor points (absolute error).
export const COLORSCALES: Record<ColorScaleName, string | [number, string][]> = {
  pressure: "Viridis",
  permeability: "Cividis",
  error: [
    [0, "#000004"],
    [0.25, "#57106e"],
    [0.5, "#bc3754"],
    [0.75, "#f98e09"],
    [1, "#fcffa4"],
  ],
};

export interface HeatmapProps {
  z: Grid;
  title: string;
  /** Label shown on the colour bar and in hover, e.g. "u" or "|û − u|". */
  valueLabel: string;
  colorscale: ColorScaleName;
  zmin?: number;
  zmax?: number;
  height?: number;
  testId?: string;
  subtitle?: string;
}

export function transpose(z: Grid): Grid {
  const n = z.length;
  const m = z[0]?.length ?? 0;
  const out: Grid = Array.from({ length: m }, () => new Array<number>(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j][i] = z[i][j];
  return out;
}

export function cellCentres(n: number): number[] {
  return Array.from({ length: n }, (_, i) => (i + 0.5) / n);
}

export function buildFigure(p: HeatmapProps) {
  const n = p.z.length;
  const coords = cellCentres(n);
  const data = [
    {
      type: "heatmap",
      z: transpose(p.z),
      x: coords,
      y: coords,
      colorscale: COLORSCALES[p.colorscale],
      zmin: p.zmin,
      zmax: p.zmax,
      zauto: p.zmin === undefined || p.zmax === undefined,
      hovertemplate: `x = %{x:.4f}<br>y = %{y:.4f}<br>${p.valueLabel} = %{z:.5g}<extra></extra>`,
      colorbar: {
        title: { text: p.valueLabel, side: "right", font: { size: 12 } },
        thickness: 12,
        outlinewidth: 0,
        tickfont: { size: 11 },
        len: 0.92,
      },
    },
  ];
  const axis = {
    range: [0, 1],
    constrain: "domain",
    tickvals: [0, 0.25, 0.5, 0.75, 1],
    ticks: "outside",
    ticklen: 3,
    tickfont: { size: 11 },
    showgrid: false,
    zeroline: false,
    mirror: true,
    showline: true,
    linecolor: "#c9ced6",
  };
  const layout = {
    autosize: true,
    height: p.height ?? 320,
    margin: { l: 44, r: 8, t: 8, b: 40 },
    xaxis: { ...axis, title: { text: "x", font: { size: 12 } } },
    yaxis: { ...axis, title: { text: "y", font: { size: 12 } }, scaleanchor: "x", scaleratio: 1 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", color: "#1d2733" },
    hoverlabel: { font: { family: "ui-monospace, SFMono-Regular, Menlo, monospace", size: 12 } },
  };
  const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"], toImageButtonOptions: { format: "png", filename: p.title.replace(/\W+/g, "_").toLowerCase(), scale: 2 } };
  return { data, layout, config };
}

export function Heatmap(props: HeatmapProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const fig = useMemo(() => buildFigure(props), [props]);
  const [lo, hi] = useMemo(() => {
    let a = Infinity;
    let b = -Infinity;
    for (const r of props.z) for (const v of r) {
      if (v < a) a = v;
      if (v > b) b = v;
    }
    return [a, b];
  }, [props.z]);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    if (!el) return;
    import("plotly.js-dist-min")
      .then(({ default: Plotly }) => {
        if (!cancelled) return Plotly.react(el, fig.data, fig.layout, fig.config);
      })
      .catch((e: unknown) => !cancelled && setFailed(e instanceof Error ? e.message : "plot failed"));
    return () => {
      cancelled = true;
    };
  }, [fig]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) import("plotly.js-dist-min").then(({ default: Plotly }) => Plotly.purge(el)).catch(() => undefined);
    };
  }, []);

  const n = props.z.length;
  return (
    <figure className="heatmap" data-testid={props.testId}>
      <figcaption className="heatmap__head">
        <span className="heatmap__title">{props.title}</span>
        {props.subtitle && <span className="heatmap__sub">{props.subtitle}</span>}
      </figcaption>
      <div
        ref={ref}
        className="heatmap__plot"
        role="img"
        aria-label={`${props.title}: ${n}×${n} heatmap, ${props.valueLabel} from ${lo.toPrecision(4)} to ${hi.toPrecision(4)}`}
        style={{ minHeight: props.height ?? 320 }}
      />
      {failed && <p className="notice notice--error">Visualisation failed to load: {failed}</p>}
      <div className="heatmap__foot mono">
        {n}×{n} · min {lo.toPrecision(4)} · max {hi.toPrecision(4)}
      </div>
    </figure>
  );
}
