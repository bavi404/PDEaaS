import type { NextConfig } from "next";

const config: NextConfig = {
  // Self-contained server bundle for the Docker image (M6).
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  // The app uses no next/image, so image optimisation is off and sharp/libvips (LGPL-3.0) is kept out of the
  // standalone server bundle (NOTICE.md, docs/DOCKER.md).
  images: { unoptimized: true },
  outputFileTracingExcludes: { "*": ["node_modules/sharp/**", "node_modules/@img/**"] },
};

export default config;
