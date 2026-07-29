"""Deploy-time data fetcher (runs in the Render build step).

The data artifacts (backtest_data.pkl etc.) are too big to commit to git, so on a
fresh host we download a prebuilt bundle from the DATA_URL env var and extract it.
If the artifacts are already present (e.g. local dev), this is a no-op.

Create the bundle locally with `python make_data_bundle.py`, upload data_bundle.zip
somewhere with a direct-download link, and set DATA_URL to it in the Render
dashboard.

PRIVATE REPO / LICENSED DATA
----------------------------
The artifacts are Sharadar-derived and Sharadar is licensed per seat, so the
bundle must NOT sit anywhere public. This repo is private, which means its
Release assets are private too -- and a private asset returns 404 to an
unauthenticated request. Set GITHUB_TOKEN in the Render environment (a
fine-grained PAT with Contents:read on this repo) and this fetcher authenticates.

GitHub's REST download for a release asset also requires
`Accept: application/octet-stream`; without it the API returns the asset's JSON
metadata, which then fails to unzip with a confusing error.
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

import config

art = config.ARTIFACT_DIR / "backtest_data.pkl"
# Version marker. "artifacts already present -> skip" is right for local dev but
# silently wrong the moment the BUNDLE changes: on any host that preserves the
# directory between deploys, a stale panel survives forever and the site keeps
# serving old data while every deploy reports success. Bump DATA_VERSION in the
# environment alongside DATA_URL and the fetcher re-downloads.
marker = config.ARTIFACT_DIR / ".data_version"
want = os.environ.get("DATA_VERSION", "").strip()
have = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
if art.exists() and (not want or want == have):
    why = "no DATA_VERSION set" if not want else f"version {have} already installed"
    print(f"[fetch_data] artifacts already present ({why}); nothing to download.")
    sys.exit(0)
if art.exists():
    print(f"[fetch_data] artifacts present but version '{have or 'unknown'}' != "
          f"requested '{want}' — re-downloading.")

url = os.environ.get("DATA_URL", "").strip()
if not url:
    print("[fetch_data] WARNING: artifacts missing and DATA_URL not set — the site "
          "will start but backtests/portfolio will be unavailable until data is provided.")
    sys.exit(0)

headers = {"User-Agent": "quant-model-deploy"}
token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("DATA_TOKEN") or "").strip()
if token:
    headers["Authorization"] = f"Bearer {token}"
    if "api.github.com" in url:
        headers["Accept"] = "application/octet-stream"

print("[fetch_data] downloading data bundle from DATA_URL ...")
req = urllib.request.Request(url, headers=headers)
tmp = None
try:
    # Stream to disk rather than into memory: the bundle is ~1GB and the Render
    # instance has 2GB total, so buffering it whole invites the OOM killer to
    # take out the build.
    with urllib.request.urlopen(req) as r, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as fh:
        tmp = fh.name
        shutil.copyfileobj(r, fh, length=1 << 22)
    size = os.path.getsize(tmp)
    print(f"[fetch_data] downloaded {size/1e6:.0f} MB; extracting to {config.ROOT} ...")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(config.ROOT)    # bundle holds data/artifacts/... and data/cache/...
except urllib.error.HTTPError as e:
    hint = ""
    if e.code in (401, 403, 404):
        hint = ("\n  -> this repo is PRIVATE, so its Release assets need auth. Set "
                "GITHUB_TOKEN (Contents:read) in the Render environment, and point "
                "DATA_URL at the api.github.com asset URL rather than the browser one.")
    print(f"[fetch_data] ERROR {e.code} fetching the bundle.{hint}", file=sys.stderr)
    sys.exit(1)
finally:
    if tmp and os.path.exists(tmp):
        os.remove(tmp)

if art.exists() and want:
    marker.write_text(want, encoding="utf-8")

ok = art.exists()
if ok:
    try:
        import json as _json
        _m = _json.loads((config.ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))
        print(f"[fetch_data] done. OK — source={_m.get('source')} "
              f"names={_m.get('n_names')} built={_m.get('built_at')}")
    except Exception:
        print("[fetch_data] done. OK")
else:
    print("[fetch_data] WARNING: backtest_data.pkl still missing — check the bundle layout.")
