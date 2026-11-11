// `output: "standalone"` does not copy static assets; the standalone server expects them next to server.js.
import { cpSync, existsSync } from "node:fs";

cpSync(".next/static", ".next/standalone/.next/static", { recursive: true });
if (existsSync("public")) cpSync("public", ".next/standalone/public", { recursive: true });
console.log("copied .next/static and public into .next/standalone");
