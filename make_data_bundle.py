"""Create data_bundle.zip for deployment.

Bundles ONLY what the live site needs: the prebuilt artifacts plus the cached
yfinance benchmark series (so the deployed app is self-contained and doesn't
depend on live yfinance at first load). Excludes the bulky fiscal.ai request
cache. Upload the resulting data_bundle.zip to a host with a direct-download link
and set DATA_URL to it on Render.

LICENCE GATE
------------
Sharadar data is licensed per-seat and MUST NOT be redistributed. This repo is
public and its Release assets are public, so a Sharadar-built artifact must
never be bundled for that path. The build refuses unless the destination is
known-private (ALLOW_LICENSED_BUNDLE=1), which you should only set once
DATA_URL points somewhere authenticated. See sharadar_kit/CLAUDE.md.
"""
from __future__ import annotations
import json
import os
import sys
import zipfile
from pathlib import Path

import config

OUT = config.ROOT / "data_bundle.zip"


def _artifact_source() -> str:
    """Which vendor built the current artifacts ('sharadar', 'fiscal', ...)."""
    p = config.ARTIFACT_DIR / "meta.json"
    if not p.exists():
        return "unknown"
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("source", "fiscal"))
    except Exception:
        return "unknown"


src = _artifact_source()
if src == "sharadar" and os.environ.get("ALLOW_LICENSED_BUNDLE") != "1":
    sys.exit(
        "REFUSING to bundle Sharadar-derived artifacts.\n"
        "  Sharadar is licensed per-seat; this repo (and its Release assets) is\n"
        "  PUBLIC, so uploading this bundle would be redistribution.\n"
        "  Point DATA_URL at an authenticated/private location first, then re-run\n"
        "  with ALLOW_LICENSED_BUNDLE=1 to confirm the destination is not public."
    )

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    # the model artifacts (factor snapshot + the backtest panel pickle)
    for p in sorted(config.ARTIFACT_DIR.glob("*")):
        # skip the kept-aside vendor backups; only ship the live artifact
        if p.is_file() and ".fiscal." not in p.name:
            z.write(p, p.relative_to(config.ROOT)); n += 1
    # cached yfinance benchmarks/gold (yf_*.pkl) so charts work without live calls
    for p in sorted(config.CACHE_DIR.glob("yf_*.pkl")):
        z.write(p, p.relative_to(config.ROOT)); n += 1

size = OUT.stat().st_size / 1e6
print(f"Wrote {OUT} ({size:.0f} MB, {n} files, source={src}).")
print("Zip layout: data/artifacts/... and data/cache/yf_*.pkl (relative to project root).")
print("Next: upload it and set DATA_URL to its download link on Render.")
