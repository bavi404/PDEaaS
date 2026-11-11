// Minimal typing for the parts of plotly.js used by components/Heatmap.tsx.
declare module "plotly.js-dist-min" {
  export interface PlotlyStatic {
    react(root: HTMLElement, data: unknown[], layout?: unknown, config?: unknown): Promise<unknown>;
    purge(root: HTMLElement): void;
    Plots: { resize(root: HTMLElement): void };
  }
  const Plotly: PlotlyStatic;
  export default Plotly;
}
