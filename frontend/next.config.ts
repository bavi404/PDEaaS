import type { NextConfig } from "next";

const config: NextConfig = {
  // Self-contained server bundle for the Docker image (M6).
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
};

export default config;
