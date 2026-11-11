#!/usr/bin/env python3
"""Deploy PDEFlow to a Hugging Face Docker Space. NOT run automatically; requires explicit authorization.

    HF_TOKEN=... uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 \
        python scripts/deploy_hf_space.py --space <owner>/<space> [--dry-run]

- The token is read ONLY from the HF_TOKEN environment variable. It is never written to disk, a git remote, logs or
  the Space, and it is never printed.
- Uploads exactly the files the Docker build needs, taken from the committed git HEAD (`git archive`), not the
  working tree. The selection is the allow-list in .dockerignore plus .dockerignore itself. The README.md is
  replaced by the Space card in deploy/huggingface/README.md (YAML front matter: sdk docker, app_port 7860).
- Refuses to run with uncommitted changes, so the Space always corresponds to a pushed commit.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
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


def allowed_paths() -> list[str]:
    pats = []
    for line in (ROOT / ".dockerignore").read_text().splitlines():
        line = line.strip()
        if line.startswith("!"):
            pats.append(line[1:].rstrip("/"))
    return pats


def selected(path: str, pats: list[str]) -> bool:
    if "__pycache__" in path or path.endswith(".pyc"):
        return False
    return any(path == p or path.startswith(p + "/") or fnmatch.fnmatch(path, p) for p in pats)


def stage(dest: Path) -> list[str]:
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("refusing to deploy: uncommitted changes (commit and push first)")
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    pats = allowed_paths() + [".dockerignore", "LICENSE", "NOTICE.md"]  # licence scope travels with the Space
    files = []
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for m in tf.getmembers():
            if m.isfile() and selected(m.name, pats):
                out = dest / m.name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tf.extractfile(m).read())
                out.chmod(m.mode)
                files.append(m.name)
    shutil.copy(ROOT / "deploy/huggingface/README.md", dest / "README.md")
    files.append("README.md (Space card)")
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<owner>/<space-name>")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage and list files only; no network, no token needed")
    ap.add_argument("--stage-to", help="with --dry-run: also copy the staged Space repository here (e.g. to docker build it)")
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                    help="after upload, poll the Space runtime stage (BUILDING -> RUNNING) for up to SECONDS")
    a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        size = sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file())
        print(f"staged {len(files)} files ({size / 1e6:.1f} MB) from commit {head[:12]}")
        for f in files:
            print("  ", f)
        if a.dry_run:
            if a.stage_to:
                shutil.copytree(tmp, a.stage_to, dirs_exist_ok=False)
                print(f"staged Space repository copied to {a.stage_to}")
            return 0
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is not set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print("authenticated as", api.whoami()["name"])
        api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
        info = api.space_info(a.space)
        print(f"space {a.space}: private={info.private}")
        if bool(info.private) != a.private:
            raise SystemExit(f"visibility mismatch: Space private={info.private}, requested private={a.private}; "
                             "change it in the Space settings, then rerun")
        api.upload_folder(repo_id=a.space, repo_type="space", folder_path=tmp, delete_patterns=["*"],
                          commit_message=f"Deploy PDEFlow from git {head[:12]}")
        t0 = time.monotonic()
        print(f"uploaded at {datetime.now(UTC).isoformat(timespec='seconds')}; the Space builds the Docker image now: "
              f"https://huggingface.co/spaces/{a.space}")
        last, rebuilt = None, False
        while a.wait and time.monotonic() - t0 < a.wait:
            rt_stage = api.get_space_runtime(a.space).stage
            if rt_stage != last:
                print(f"  t+{time.monotonic() - t0:6.0f}s  stage {rt_stage}", flush=True)
                last = rt_stage
            # RUNNING only counts after this upload was seen building/starting (a redeploy reports the OLD
            # version as RUNNING / RUNNING_BUILDING until the new container takes over)
            rebuilt |= rt_stage in ("BUILDING", "APP_STARTING", "RUNNING_BUILDING", "RUNNING_APP_STARTING")
            if rt_stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
                return 1
            if rt_stage == "RUNNING" and rebuilt:
                return 0
            time.sleep(10)
        if a.wait:
            print(f"  still {last} after {a.wait}s")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
