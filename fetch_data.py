"""Deploy-time data fetcher (runs in the Render build step).

The data artifacts (backtest_data.pkl etc.) are too big to commit to git, so on a
fresh host we download a prebuilt bundle from the DATA_URL env var and extract it.
If the artifacts are already present (e.g. local dev), this is a no-op.

Create the bundle locally with `python make_data_bundle.py`, upload data_bundle.zip
somewhere with a direct-download link (a GitHub Release asset is easiest), and set
DATA_URL to that link in the Render dashboard.
"""
from __future__ import annotations
import io
import os
import sys
import urllib.request
import zipfile

import config

art = config.ARTIFACT_DIR / "backtest_data.pkl"
if art.exists():
    print(f"[fetch_data] artifacts already present ({art}); nothing to download.")
    sys.exit(0)

url = os.environ.get("DATA_URL", "").strip()
if not url:
    print("[fetch_data] WARNING: artifacts missing and DATA_URL not set — the site "
          "will start but backtests/portfolio will be unavailable until data is provided.")
    sys.exit(0)

print(f"[fetch_data] downloading data bundle from DATA_URL ...")
req = urllib.request.Request(url, headers={"User-Agent": "quant-model-deploy"})
with urllib.request.urlopen(req) as r:
    data = r.read()
print(f"[fetch_data] downloaded {len(data)/1e6:.0f} MB; extracting to {config.ROOT} ...")
with zipfile.ZipFile(io.BytesIO(data)) as z:
    z.extractall(config.ROOT)        # bundle contains data/artifacts/... and data/cache/...
print("[fetch_data] done.", "OK" if art.exists() else "WARNING: backtest_data.pkl still missing — check the bundle layout.")
