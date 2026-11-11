#!/usr/bin/env python3
"""Stage (and, only with explicit owner approval, deploy) the ZeroGPU Gradio Space (docs/ZEROGPU.md §6).

    # stage + list files; no network, no token:
    .venv/bin/python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --dry-run [--stage-to DIR]
    # deploy (requires HF_TOKEN in the environment and the owner's explicit approval):
    uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 \
        python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --wait 2400

Space repository layout (app.py finds backend/ next to itself):
    app.py  requirements.txt  README.md (Space card: sdk gradio, python 3.12.12)  LICENSE  NOTICE.md
    backend/pdeflow/**  backend/models/fno_darcy_beta1_baseline.pt  backend/assets/demo_samples_v1.npz
    reports/m2_eval_fno_baseline.json  web/** (static export of the frontend)

Guards:
- committed files only (`git archive HEAD`), and a clean tree
- the web export must come from the committed frontend tree
- the checkpoint SHA-256 must equal the pinned value
- HF_TOKEN is read from the environment only and never printed
- hardware ZeroGPU (`zero-a10g`, free for eligible personal accounts); public visibility is verified
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA = "d7133c7c681cf3e9632579833fe357636d84ff7a5678d94b2f24c38955124df7"
MAP = {  # Space path -> repository path (from git HEAD)
    "app.py": "deploy/zerogpu/app.py",
    "requirements.txt": "deploy/zerogpu/requirements.txt",
    "README.md": "deploy/zerogpu/README.md",
    "LICENSE": "LICENSE",
    "NOTICE.md": "NOTICE.md",
    "backend/models/fno_darcy_beta1_baseline.pt": "backend/models/fno_darcy_beta1_baseline.pt",
    "backend/assets/demo_samples_v1.npz": "backend/assets/demo_samples_v1.npz",
    "reports/m2_eval_fno_baseline.json": "reports/m2_eval_fno_baseline.json",
}
PACKAGE_PREFIX = "backend/pdeflow/"


def git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


def stage(dest: Path, web: Path) -> list[str]:
    if git("status", "--porcelain"):
        raise SystemExit("refusing: uncommitted changes (commit and push first)")
    info = json.loads((web / "build-info.json").read_text())
    if info.get("frontend_dirty") or info.get("frontend_tree") != git("rev-parse", "HEAD:frontend"):
        raise SystemExit("refusing: web export was not built from the committed frontend; rerun deploy/zerogpu/build_frontend.py")
    wanted = {v: k for k, v in MAP.items()}
    files = []
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", "HEAD", binary=True))) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            target = wanted.get(m.name)
            if target is None and m.name.startswith(PACKAGE_PREFIX) and "__pycache__" not in m.name:
                target = m.name
            if target is None:
                continue
            out = dest / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tf.extractfile(m).read())
            files.append(target)
    missing = set(MAP) - set(files)
    if missing:
        raise SystemExit(f"refusing: missing from HEAD: {sorted(missing)}")
    sha = hashlib.sha256((dest / "backend/models/fno_darcy_beta1_baseline.pt").read_bytes()).hexdigest()
    if sha != PINNED_SHA:
        raise SystemExit(f"refusing: checkpoint sha256 {sha} != pinned {PINNED_SHA}")
    shutil.copytree(web, dest / "web")
    files += [f"web/{p.relative_to(web).as_posix()}" for p in sorted(web.rglob("*")) if p.is_file()]
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--web", default=str(ROOT / "deploy/zerogpu/web"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage-to")
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS")
    a = ap.parse_args()
    head = git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp), Path(a.web))
        size = sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file())
        n_web = sum(f.startswith("web/") for f in files)
        print(f"staged {len(files)} files ({size / 1e6:.1f} MB; {n_web} web files) from commit {head[:12]}; "
              f"checkpoint sha256 {PINNED_SHA[:12]}… verified")
        for f in files:
            if not f.startswith("web/"):
                print("  ", f)
        print(f"   web/… ({n_web} files)")
        if a.dry_run:
            if a.stage_to:
                shutil.copytree(tmp, a.stage_to)
                print(f"staged Space repository copied to {a.stage_to}")
            return 0
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is not set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print("authenticated as", api.whoami()["name"])
        api.create_repo(a.space, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", private=False,
                        exist_ok=True)
        info = api.space_info(a.space)
        print(f"space {a.space}: private={info.private}")
        if info.private:
            raise SystemExit("refusing: Space is private; the brief requires a public Space")
        api.upload_folder(repo_id=a.space, repo_type="space", folder_path=tmp, delete_patterns=["*"],
                          commit_message=f"Deploy PDEFlow (ZeroGPU/Gradio) from git {head[:12]}")
        t0 = time.monotonic()
        print(f"uploaded at {datetime.now(UTC).isoformat(timespec='seconds')}: https://huggingface.co/spaces/{a.space}")
        last, rebuilt = None, False
        while a.wait and time.monotonic() - t0 < a.wait:
            rt = api.get_space_runtime(a.space)
            if rt.stage != last:
                print(f"  t+{time.monotonic() - t0:6.0f}s  stage {rt.stage}  hardware {rt.hardware}", flush=True)
                last = rt.stage
            rebuilt |= rt.stage in ("BUILDING", "APP_STARTING", "RUNNING_BUILDING", "RUNNING_APP_STARTING")
            if rt.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
                return 1
            if rt.stage == "RUNNING" and rebuilt:
                return 0
            time.sleep(10)
        if a.wait:
            print(f"  still {last} after {a.wait}s")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
