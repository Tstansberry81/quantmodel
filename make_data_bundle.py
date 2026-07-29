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
import json as _json
import os
import pathlib
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

# Ship ONLY the live artifact. Every kept-aside vendor/backup copy
# (backtest_data.fiscal.pkl, .large.pkl, .sharadar_v1.pkl, meta.fiscal.json)
# lives in the same directory, and a bare glob swept them all in -- the first
# bundle came out at 1.2GB because it carried three redundant panels.
_BACKUP_MARKERS = (".fiscal.", ".large.", ".sharadar_v1.", ".accel-archive.")


def _is_backup(name: str) -> bool:
    return any(m in name for m in _BACKUP_MARKERS)


# Deploy-trim: the shipped model floors at $10B market cap, so a name that
# NEVER reached DEPLOY_MCAP_KEEP can never be selected and only costs the host
# memory and download time. Trimming well below the floor keeps the universe
# ranking identical for everything the model can actually pick.
DEPLOY_MCAP_KEEP = 5e9
_trimmed = None
_live = config.ARTIFACT_DIR / "backtest_data.pkl"
if _live.exists():
    import pickle
    import tempfile
    with open(_live, "rb") as f:
        _data = pickle.load(f)
    _keep = {}
    for _ck, _b in _data.items():
        _fh = _b.get("fund_hist")
        try:
            _mx = float(_fh["calculated_market_cap"].max()) if (
                _fh is not None and "calculated_market_cap" in getattr(_fh, "columns", [])
            ) else 0.0
        except Exception:
            _mx = 0.0
        if _mx >= DEPLOY_MCAP_KEEP:
            _keep[_ck] = _b
    _trimmed = pathlib.Path(tempfile.mkstemp(suffix=".pkl")[1])
    with open(_trimmed, "wb") as f:
        pickle.dump(_keep, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"deploy-trim: {len(_data):,} names -> {len(_keep):,} that ever reached "
          f"${DEPLOY_MCAP_KEEP/1e9:.0f}B "
          f"({_live.stat().st_size/1e6:.0f}MB -> {_trimmed.stat().st_size/1e6:.0f}MB)")

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    # the model artifacts (factor snapshot + the backtest panel pickle)
    for p in sorted(config.ARTIFACT_DIR.glob("*")):
        if not p.is_file() or _is_backup(p.name):
            continue
        if p.name == "backtest_data.pkl" and _trimmed is not None:
            z.write(_trimmed, p.relative_to(config.ROOT)); n += 1
            continue
        if p.name == "meta.json" and _trimmed is not None:
            # Rewrite the count to match what actually ships. Copying the
            # untrimmed meta made the live footer claim 12,164 names for a
            # bundle carrying 2,976 -- a number the deployed site displays.
            _mj = _json.loads(p.read_text(encoding="utf-8"))
            _mj["n_names"] = _mj["universe_size"] = len(_keep)
            _mj["deploy_trim_mcap"] = DEPLOY_MCAP_KEEP
            z.writestr(str(p.relative_to(config.ROOT)), _json.dumps(_mj)); n += 1
            continue
        z.write(p, p.relative_to(config.ROOT)); n += 1
    # cached yfinance benchmarks/gold (yf_*.pkl) so charts work without live calls
    for p in sorted(config.CACHE_DIR.glob("yf_*.pkl")):
        z.write(p, p.relative_to(config.ROOT)); n += 1

if _trimmed is not None and _trimmed.exists():
    _trimmed.unlink()

size = OUT.stat().st_size / 1e6
print(f"Wrote {OUT} ({size:.0f} MB, {n} files, source={src}).")
print("Zip layout: data/artifacts/... and data/cache/yf_*.pkl (relative to project root).")
print("Next: upload it and set DATA_URL to its download link on Render.")
