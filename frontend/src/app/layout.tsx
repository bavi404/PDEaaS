import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { AppShell } from "@/components/AppShell";
import { PredictionProvider } from "@/lib/store";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "PDEFlow", template: "%s · PDEFlow" },
  description: "Fourier Neural Operator surrogate for 2D Darcy flow, trained on PDEBench. PDE solutions as a service.",
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#1f4e79" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <PredictionProvider>
          <AppShell>{children}</AppShell>
        </PredictionProvider>
      </body>
    </html>
  );
}
