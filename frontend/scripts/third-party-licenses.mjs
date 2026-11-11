// Generates public/third-party-licenses.txt: the licence texts of the npm packages installed for production
// (package-lock entries without "dev"), so the notices required by MIT/BSD/Apache licences travel with every copy
// of the built frontend. Minifiers strip licence comments such as Plotly's MIT header. Runs before `next build`
// (npm run build, Docker build, deploy/zerogpu/build_frontend.py). The output is generated and not committed.
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const lock = JSON.parse(readFileSync("package-lock.json", "utf8"));
// present in node_modules at build time but NOT shipped: removed from the server trace (next.config.ts) or build-time only
const NOT_SHIPPED = [/^sharp$/, /^@img\//, /^@next\/swc-/];

const entries = Object.entries(lock.packages)
  .filter(([k, v]) => k && !v.dev && existsSync(k))
  .map(([k, v]) => {
    const name = k.replace(/^.*node_modules\//, "");
    const pkg = JSON.parse(readFileSync(join(k, "package.json"), "utf8"));
    const repo = typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url ?? pkg.homepage ?? "";
    const files = readdirSync(k).filter((f) => /^(licen[cs]e|copying|notice)/i.test(f)).sort();
    // a leading /*! ... */ or /** ... */ licence header of the main file (e.g. plotly.min.js), which minifiers strip
    const main = join(k, typeof pkg.main === "string" ? pkg.main : "index.js");
    let header = "";
    if (existsSync(main)) {
      const head = readFileSync(main, "utf8").slice(0, 2000).trimStart();
      const m = head.match(/^\/\*[\s\S]*?\*\//);
      if (m && /copyright|licen[cs]e/i.test(m[0])) header = m[0];
    }
    return { name, version: v.version, license: v.license ?? pkg.license ?? "UNKNOWN", repo, files, dir: k, header };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

const shipped = entries.filter((e) => !NOT_SHIPPED.some((r) => r.test(e.name)));
const out = [
  "THIRD-PARTY SOFTWARE NOTICES: PDEFlow web frontend",
  "",
  "The PDEFlow frontend bundles the npm packages listed below. Each keeps its own licence; PDEFlow's MIT licence does",
  "not apply to them (see NOTICE.md in the repository). The minified bundles may not contain the original licence",
  "comments, so the full licence texts are reproduced here.",
  "",
  `Packages: ${shipped.length}. Generated from frontend/package-lock.json by scripts/third-party-licenses.mjs.`,
  "Not shipped and therefore not listed: sharp and @img/* (libvips, LGPL-3.0; removed from the server trace,",
  "image optimisation is disabled) and @next/swc-* (build-time compiler).",
  "",
];
for (const e of shipped) {
  out.push("=".repeat(100), `${e.name}@${e.version}  |  License: ${e.license}${e.repo ? `  |  ${e.repo}` : ""}`, "=".repeat(100));
  if (e.files.length === 0) {
    out.push(`(The npm package contains no licence file. It is declared "${e.license}" in its package.json; see the source repository.)`);
  }
  if (e.header) out.push("--- licence header of the bundled file ---", e.header);
  for (const f of e.files) out.push(`--- ${f} ---`, readFileSync(join(e.dir, f), "utf8").trimEnd());
  out.push("");
}
writeFileSync("public/third-party-licenses.txt", out.join("\n") + "\n");
console.log(`third-party-licenses.txt: ${shipped.length} packages (${entries.length - shipped.length} not shipped, not listed)`);
