"""Create data_bundle.zip for deployment.

Bundles ONLY what the live site needs: the prebuilt artifacts plus the cached
yfinance benchmark series (so the deployed app is self-contained and doesn't
depend on live yfinance at first load). Excludes the bulky fiscal.ai request
cache. Upload the resulting data_bundle.zip to a host with a direct-download link
(a GitHub Release asset is simplest) and set DATA_URL to it on Render.
"""
from __future__ import annotations
import zipfile
from pathlib import Path

import config

OUT = config.ROOT / "data_bundle.zip"
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    # the model artifacts (factor snapshot + the backtest panel pickle)
    for p in sorted(config.ARTIFACT_DIR.glob("*")):
        if p.is_file():
            z.write(p, p.relative_to(config.ROOT)); n += 1
    # cached yfinance benchmarks/gold (yf_*.pkl) so charts work without live calls
    for p in sorted(config.CACHE_DIR.glob("yf_*.pkl")):
        z.write(p, p.relative_to(config.ROOT)); n += 1

size = OUT.stat().st_size / 1e6
print(f"Wrote {OUT} ({size:.0f} MB, {n} files).")
print("Zip layout: data/artifacts/... and data/cache/yf_*.pkl (relative to project root).")
print("Next: upload it (e.g. a GitHub Release asset) and set DATA_URL to its download link on Render.")
