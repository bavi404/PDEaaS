#!/usr/bin/env python3
"""Pre-deployment check: does Hugging Face's Gradio-SDK build install command resolve with our requirements?

The Gradio SDK build does not run `pip install -r requirements.txt` alone. It adds its own packages in ONE pip
command. This is the form observed in the build log of deployment attempt 1 (2026-11-11, Space bavi404/pdeflow,
docs/ZEROGPU.md §7):

    pip install --no-cache-dir -r requirements.txt "torch<=2.13.0" gradio[oauth,mcp]==<sdk_version> \\
        "uvicorn>=0.14.0" "websockets>=10.4" spaces==0.51.3

Attempt 1 failed with ResolutionImpossible: gradio[mcp]==6.30.0 needs pydantic<=2.12.5. This check reproduces the
exact command with **pip** (--dry-run, in a throwaway Python 3.12 venv; only metadata is downloaded) and cross-checks
with uv. It exits non-zero if either fails to resolve. No network calls to Hugging Face.

    python deploy/zerogpu/check_hf_build.py [--out report.txt]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQ = HERE / "requirements.txt"
CARD = HERE / "README.md"
PLATFORM_SPACES = "spaces==0.51.3"  # pinned by the platform in the observed build command


def hf_build_args() -> list[str]:
    sdk = re.search(r"^sdk_version:\s*(\S+)", CARD.read_text(), re.MULTILINE).group(1)
    return ["-r", str(REQ), "torch<=2.13.0", f"gradio[oauth,mcp]=={sdk}", "uvicorn>=0.14.0", "websockets>=10.4",
            PLATFORM_SPACES]


def run(cmd: list[str], env: dict) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False, timeout=3600)
    return p.returncode, (p.stdout + p.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--python", default="3.12.12")
    a = ap.parse_args()
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    args = hf_build_args()
    lines = ["HF Gradio-SDK build command (reproduced):", "  pip install " + " ".join(a if " " not in a else repr(a) for a in args), ""]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        code, out = run(["uv", "venv", "-q", "--seed", "--python", a.python, str(venv)], env)
        if code:
            raise SystemExit(f"could not create a Python {a.python} venv: {out}")
        code, out = run([str(venv / "bin/python"), "-m", "pip", "install", "--dry-run", "--quiet", "--report",
                         str(Path(tmp) / "report.json"), *args], env)
        pip_ok = code == 0
        ok &= pip_ok
        resolved = {}
        if pip_ok:
            import json

            rep = json.loads((Path(tmp) / "report.json").read_text())
            resolved = {i["metadata"]["name"].lower(): i["metadata"]["version"] for i in rep["install"]}
        lines += [f"pip --dry-run (Python {a.python}): {'RESOLVES' if pip_ok else 'FAILS'}"
                  + (f", {len(resolved)} packages" if pip_ok else ""), *(["  " + ln for ln in out.strip().splitlines()[-25:]] if not pip_ok else [])]
        for name in ("torch", "gradio", "spaces", "fastapi", "starlette", "pydantic", "pydantic-core", "uvicorn",
                     "numpy", "mcp", "neuraloperator"):
            if name in resolved:
                lines.append(f"  {name}=={resolved[name]}")
        # resolver cross-check: uv on the same requirement set
        inp = Path(tmp) / "hf_build.in"
        inp.write_text("\n".join(args[2:] + [f"-r {REQ}"]) + "\n")
        code, out = run(["uv", "pip", "compile", str(inp), "--python-version", a.python.rsplit(".", 1)[0],
                         "--python-platform", "x86_64-unknown-linux-gnu", "--no-header", "-o", os.devnull], env)
        uv_ok = code == 0
        ok &= uv_ok
        lines.append(f"uv pip compile cross-check: {'RESOLVES' if uv_ok else 'FAILS'}")
        if not uv_ok:
            lines += ["  " + ln for ln in out.strip().splitlines()[-15:]]
    # every pin in our lock must be what pip would install (no silent up/downgrades by the build)
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-\[\]]+)==([^\s;#]+)", REQ.read_text(), re.MULTILINE))
    pins = {re.sub(r"\[.*\]", "", k).lower().replace("_", "-"): v for k, v in pins.items()}
    drift = {k: (v, resolved.get(k)) for k, v in pins.items() if k in resolved and resolved[k].split("+")[0] != v.split("+")[0]}
    if resolved:
        lines.append(f"pins changed by the build resolution: {drift or 'none'}")
        ok &= not drift
    lines.append("RESULT: " + ("PASS (HF build command resolves to our pinned versions)" if ok else "FAIL"))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
