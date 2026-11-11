#!/usr/bin/env python3
"""Download one PDEBench 2D Darcy Flow file from DaRUS and verify it against the official listing.

Mechanism (same as PDEBench's own downloader): resolve the file in the DaRUS Dataverse
dataset doi:10.18419/darus-2986, then GET /api/access/datafile/{id}, which redirects (HTTP 303)
to pre-signed object storage. The expected size and MD5 come from the live Dataverse file
listing, not from hard-coded values, and a provenance record is written next to the file.

Standard library only (runs before the project venv exists).

Usage:
    python scripts/download_pdebench.py --beta 1.0 --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402

DARUS = "https://darus.uni-stuttgart.de"
DOI = "doi:10.18419/darus-2986"
CHUNK = 8 * 1024 * 1024


def fetch_listing() -> list[dict]:
    url = f"{DARUS}/api/datasets/:persistentId/versions/:latest/files?persistentId={DOI}"
    with urllib.request.urlopen(url, timeout=60) as r:
        body = json.load(r)
    if body.get("status") != "OK":
        raise RuntimeError(f"DaRUS listing failed: {body.get('status')}")
    return body["data"]


def find_file(listing: list[dict], filename: str) -> dict:
    matches = [f for f in listing if f["dataFile"]["filename"] == filename]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one DaRUS entry for {filename}, found {len(matches)}")
    return matches[0]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", default="1.0", help="forcing value as written in the PDEBench filename")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--log", default="logs/m1_download.jsonl")
    args = ap.parse_args()

    log = get_logger("pdeflow.download", args.log)
    filename = f"2D_DarcyFlow_beta{args.beta}_Train.hdf5"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename
    part = dest.with_suffix(dest.suffix + ".part")

    listing = fetch_listing()
    entry = find_file(listing, filename)
    df = entry["dataFile"]
    expected_size = int(df["filesize"])
    checksum = df.get("checksum", {})
    if checksum.get("type") != "MD5":
        raise RuntimeError(f"unexpected checksum type in listing: {checksum}")
    expected_md5 = checksum["value"]
    file_url = f"{DARUS}/api/access/datafile/{df['id']}"
    log_event(log, "listing_resolved", filename=filename, darus_file_id=df["id"],
              directory=entry.get("directoryLabel"), expected_size=expected_size,
              expected_md5=expected_md5, url=file_url, restricted=entry.get("restricted"))

    if dest.exists() and dest.stat().st_size == expected_size:
        log_event(log, "file_exists_reverifying", path=str(dest))
    else:
        t0 = time.monotonic()
        h = hashlib.md5()
        done, next_report = 0, 0
        req = urllib.request.Request(file_url, headers={"User-Agent": "pdeflow-downloader/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
            log_event(log, "download_started", http_status=r.status,
                      content_length=r.headers.get("Content-Length"),
                      final_host=urllib.parse.urlsplit(r.geturl()).netloc)
            while chunk := r.read(CHUNK):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if done >= next_report or done == expected_size:
                    el = time.monotonic() - t0
                    log_event(log, "download_progress", bytes=done, total=expected_size,
                              pct=round(100 * done / expected_size, 1), elapsed_s=round(el, 1),
                              mb_per_s=round(done / 1e6 / max(el, 1e-9), 2))
                    next_report += 10 * expected_size // 100
        elapsed = time.monotonic() - t0
        if done != expected_size:
            log_event(log, "size_mismatch", 40, got=done, expected=expected_size)
            return 1
        streamed_md5 = h.hexdigest()
        log_event(log, "download_finished", bytes=done, elapsed_s=round(elapsed, 1), streamed_md5=streamed_md5)
        part.rename(dest)

    # Independent re-hash of the file on disk (does not trust the streaming hash).
    t0 = time.monotonic()
    disk_md5 = md5_of(dest)
    ok = disk_md5 == expected_md5 and dest.stat().st_size == expected_size
    log_event(log, "checksum_verified" if ok else "checksum_mismatch", 20 if ok else 40,
              path=str(dest), size=dest.stat().st_size, md5_disk=disk_md5, md5_darus=expected_md5,
              hash_s=round(time.monotonic() - t0, 1))
    if not ok:
        return 1

    sha256 = hashlib.sha256()
    with dest.open("rb") as f:
        while chunk := f.read(CHUNK):
            sha256.update(chunk)
    prov_path = out_dir / (filename + ".provenance.json")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    retrieved, retrieved_src = now, "this run"
    if prov_path.exists():  # re-verification: keep the original retrieval time
        old = json.loads(prov_path.read_text())
        retrieved = old.get("retrieved_utc", now)
        retrieved_src = old.get("retrieved_utc_source", "earlier run")
    prov = {
        "dataset": "PDEBench", "dataset_doi": DOI, "directory": entry.get("directoryLabel"),
        "filename": filename, "darus_file_id": df["id"], "url": file_url,
        "size_bytes": expected_size, "md5": disk_md5, "md5_source": "DaRUS Dataverse file listing",
        "sha256": sha256.hexdigest(),
        "retrieved_utc": retrieved,
        "retrieved_utc_source": retrieved_src,
        "last_verified_utc": now,
        "license_note": "See DaRUS dataset page for the dataset licence; data is not redistributed in this repo.",
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    log_event(log, "provenance_written", **prov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
