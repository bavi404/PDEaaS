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
    pats = allowed_paths() + [".dockerignore"]
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
    a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        size = sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file())
        print(f"staged {len(files)} files ({size / 1e6:.1f} MB) from commit {head[:12]}")
        for f in files:
            print("  ", f)
        if a.dry_run:
            return 0
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is not set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print("authenticated as", api.whoami()["name"])
        api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=a.space, repo_type="space", folder_path=tmp, delete_patterns=["*"],
                          commit_message=f"Deploy PDEFlow from git {head[:12]}")
        print(f"uploaded; the Space builds the Docker image now: https://huggingface.co/spaces/{a.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
