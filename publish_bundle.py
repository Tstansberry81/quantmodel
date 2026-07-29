"""Publish data_bundle.zip to the GitHub release under a FIXED asset name.

    python make_data_bundle.py      # build it  (needs ALLOW_LICENSED_BUNDLE=1)
    python publish_bundle.py        # ship it   (no environment edits, ever)

WHY THIS EXISTS. Every upload used to create a new release asset with a new
numeric id, so DATA_URL had to be re-pointed by hand -- and because the fetcher
skips when artifacts are already present, DATA_VERSION had to be bumped by hand
too. Two manual environment edits per data change, each of which silently serves
stale numbers if forgotten.

This uploads under one constant name (data_bundle.zip), deleting any existing
asset of that name first, so DATA_URL can point at the release TAG forever and
fetch_data.py resolves the current asset underneath it. The version is derived
from the asset's id + updated_at, so a republish is detected automatically.

Credentials are read from the git credential helper (macOS keychain) or
GITHUB_TOKEN, and are never echoed or written to disk.
"""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import sys

try:
    import requests
except ImportError:                                     # pragma: no cover
    sys.exit("publish_bundle: `pip install requests` first.")

REPO = os.environ.get("DATA_REPO", "Tstansberry81/quantmodel")
TAG = os.environ.get("DATA_TAG", "data-v2-sharadar")
ASSET = os.environ.get("DATA_ASSET_NAME", "data_bundle.zip")
BUNDLE = pathlib.Path(__file__).parent / "data_bundle.zip"


def _token() -> str:
    """Env first, then the git credential helper. Never printed."""
    tok = (os.environ.get("GITHUB_TOKEN") or os.environ.get("DATA_TOKEN") or "").strip()
    if tok:
        return tok
    try:
        out = subprocess.run(["git", "credential", "fill"],
                             input="protocol=https\nhost=github.com\n\n",
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    sys.exit("publish_bundle: no credential found (set GITHUB_TOKEN or run `git credential fill`).")


def main() -> None:
    if not BUNDLE.exists():
        sys.exit(f"publish_bundle: {BUNDLE} not found — run make_data_bundle.py first.")
    tok = _token()
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    base = f"https://api.github.com/repos/{REPO}"

    r = requests.get(f"{base}/releases/tags/{TAG}", headers=h, timeout=60)
    if r.status_code != 200:
        sys.exit(f"publish_bundle: release '{TAG}' lookup failed ({r.status_code}): "
                 f"{r.json().get('message', '')}")
    rel = r.json()

    # UPLOAD FIRST, THEN SWAP. Deleting the live asset before uploading its
    # replacement opens a window where the release has no asset of this name at
    # all -- and because DATA_URL now resolves by name instead of pinning an id,
    # any deploy landing in that window fails outright with "release has no asset
    # named ...". A 300MB upload over a flaky link is exactly when that happens.
    # GitHub rejects a duplicate name, so upload under a staging name, delete the
    # old one only once the bytes are safely up, then PATCH the name into place.
    staging = f"{ASSET}.incoming"
    for a in rel.get("assets", []):
        if a.get("name") == staging:            # leftover from an earlier failure
            requests.delete(f"{base}/releases/assets/{a['id']}", headers=h, timeout=60)

    size_mb = BUNDLE.stat().st_size / 1e6
    print(f"  uploading {size_mb:.0f} MB as {staging} ...")
    with open(BUNDLE, "rb") as fh:
        up = requests.post(rel["upload_url"].split("{")[0], params={"name": staging},
                           headers={**h, "Content-Type": "application/zip"},
                           data=fh, timeout=3600)
    if up.status_code not in (200, 201):
        sys.exit(f"publish_bundle: upload failed ({up.status_code}): {up.text[:300]}\n"
                 f"  the previous {ASSET} is untouched and still serving.")
    new_id = up.json()["id"]

    for a in rel.get("assets", []):
        if a.get("name") == ASSET:
            d = requests.delete(f"{base}/releases/assets/{a['id']}", headers=h, timeout=60)
            print(f"  retired previous asset {a['id']} ({d.status_code})")

    ren = requests.patch(f"{base}/releases/assets/{new_id}", headers=h,
                         json={"name": ASSET}, timeout=60)
    if ren.status_code != 200:
        sys.exit(f"publish_bundle: uploaded OK but rename failed ({ren.status_code}). "
                 f"Asset {new_id} is on the release as '{staging}' — rename it to "
                 f"'{ASSET}' by hand, or re-run.")
    a = ren.json()
    print(f"  done: asset {a['id']} · {a.get('state')} · {a['size']/1e6:.0f} MB")
    print()
    print("Nothing to change in the Render environment. It should already hold:")
    print(f"  DATA_URL = https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    print("  (no DATA_VERSION needed — the fetcher versions off the asset itself)")


if __name__ == "__main__":
    main()
