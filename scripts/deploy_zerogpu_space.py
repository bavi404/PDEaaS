#!/usr/bin/env python3
"""Stage (and, only with explicit owner approval, deploy) the ZeroGPU Gradio Space (docs/ZEROGPU.md §6).

    # stage + list files; no network:
    .venv/bin/python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --dry-run [--stage-to DIR]
    # read-only preflight of the existing Space (authentication + owner/SDK/visibility/hardware); uploads nothing:
    uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 \
        python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --preflight-only
    # deploy (owner's explicit approval only); the Space must already exist (created by the owner on the website):
    uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 \
        python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --confirm-deploy bavi404/pdeflow --wait 2400

Authentication: host-scoped proxy secret. The sandbox proxy adds the Hugging Face credential to requests for
huggingface.co. This script never reads, stores or prints a token: identity comes from a token-less
GET /api/whoami-v2, and every Hub call uses HfApi(token=False), so no local token or HF_TOKEN is ever sent; the proxy
supplies the credential. Uploads fetch a short-lived Xet/LFS token from huggingface.co (also proxy-authenticated)
and use it on the storage hosts. The script never creates a Space.

Space repository layout (app.py finds backend/ next to itself):
    app.py  requirements.txt  README.md (Space card: sdk gradio, python 3.12.12)  LICENSE  NOTICE.md
    backend/pdeflow/**  backend/models/fno_darcy_beta1_baseline.pt  backend/assets/demo_samples_v1.npz
    reports/m2_eval_fno_baseline.json  web/** (static export of the frontend)

Guards:
- committed files only (`git archive HEAD`), and a clean tree
- the web export must come from the committed frontend tree
- the checkpoint SHA-256 must equal the pinned value
- no token is read, stored or printed (proxy authentication only)
- preflight, before any upload: the Space exists; its owner equals the authenticated account and the requested
  owner; sdk gradio; public; hardware ZeroGPU (zero-*)
- upload only with --confirm-deploy equal to --space
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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


class PreflightError(SystemExit):
    pass


def summarize_whoami(d: dict) -> dict:
    """Only non-secret identity fields from a whoami-v2 response (the response never contains the token value)."""
    tok = (d.get("auth") or {}).get("accessToken") or {}
    return {"name": d.get("name"), "type": d.get("type"), "role": tok.get("role")}


def whoami_via_proxy(endpoint: str = "https://huggingface.co") -> dict:
    """GET /api/whoami-v2 WITHOUT an Authorization header; the sandbox proxy adds the host-scoped credential."""
    from huggingface_hub.utils import get_session

    r = get_session().get(f"{endpoint.rstrip('/')}/api/whoami-v2", headers={"User-Agent": "pdeflow-deploy"}, timeout=30)
    if r.status_code != 200:
        raise PreflightError(f"refusing: proxy-authenticated whoami-v2 returned HTTP {r.status_code}")
    me = summarize_whoami(r.json())
    if not me["name"]:
        raise PreflightError("refusing: whoami-v2 returned no account name")
    return me


def check_space(space: str, account: str, *, author: str | None, sdk: str | None, private: bool | None,
                hardware: str | None, requested_hardware: str | None) -> dict:
    """Pure preflight rules (unit-tested). Returns a summary or raises PreflightError."""
    owner = space.split("/", 1)[0]
    problems = []
    if owner != account:
        problems.append(f"requested owner {owner!r} != authenticated account {account!r}")
    if author != owner:
        problems.append(f"Space author {author!r} != {owner!r}")
    if sdk != "gradio":
        problems.append(f"sdk {sdk!r} != 'gradio' (ZeroGPU requires the Gradio SDK)")
    if private is not False:
        problems.append(f"visibility private={private!r}; must be public")
    hw = hardware or requested_hardware
    if not (hw or "").startswith("zero-"):
        problems.append(f"hardware {hw!r} is not ZeroGPU (zero-*); select ZeroGPU in the Space settings")
    if problems:
        raise PreflightError("refusing (preflight): " + "; ".join(problems))
    return {"space": space, "author": author, "sdk": sdk, "private": private, "hardware": hw}


def preflight(api, space: str, account: str) -> dict:
    """Read-only checks of the EXISTING Space before any upload. Never creates or modifies it."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = api.space_info(space)
    except HfHubHTTPError as e:  # RepositoryNotFoundError is a subclass; versions differ in how a 404 is classified
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403, 404):
            raise PreflightError(f"refusing: Space {space} does not exist or is not accessible (HTTP {code}); create it "
                                 "on the website (public, Gradio SDK, ZeroGPU hardware) first. This script never "
                                 "creates Spaces.") from None
        raise
    rt = info.runtime
    summary = check_space(space, account, author=info.author, sdk=info.sdk, private=info.private,
                          hardware=getattr(rt, "hardware", None), requested_hardware=getattr(rt, "requested_hardware", None))
    print(f"preflight OK: {summary}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--web", default=str(ROOT / "deploy/zerogpu/web"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage-to")
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS")
    ap.add_argument("--preflight-only", action="store_true", help="authenticate and check the Space; upload nothing")
    ap.add_argument("--confirm-deploy", metavar="OWNER/SPACE", help="must equal --space to allow the upload")
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
        from huggingface_hub import HfApi

        api = HfApi(token=False)  # never send a local token; the proxy authenticates huggingface.co requests
        me = whoami_via_proxy(api.endpoint)
        print(f"authenticated via proxy: account {me['name']} ({me['type']}), token role {me['role']}")
        info = preflight(api, a.space, me["name"])
        if a.preflight_only:
            print("preflight passed; --preflight-only: nothing uploaded")
            return 0
        if a.confirm_deploy != a.space:
            raise SystemExit(f"refusing to upload: pass --confirm-deploy {a.space} (owner approval required)")
        print(f"uploading to {a.space} (sdk {info['sdk']}, hardware {info['hardware']})")
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
