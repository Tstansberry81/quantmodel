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
import shutil
import sys
import tempfile
import time
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


# Deploy-trim: the shipped model floors at $10B market cap (EDGE_SPEC
# mcap_floor), so a name that NEVER reached that can never be selected -- it
# only costs the host memory, download time and, critically, panel-build TIME:
# daily_return_matrix is (trading days x names), so every surplus name is a
# column of ~9,700 floats rebuilt on every cold start. Render's proxy times out
# around 100s, so build cost is the binding constraint, not disk.
#
# Trimming AT the floor is safe for selection: load_edge_panel takes the top-N
# by point-in-time market cap and then applies the floor, so dropping names that
# never cleared the floor cannot change which names survive it.
DEPLOY_MCAP_KEEP = 1e10
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

# ---- precompute the panel the host would otherwise build on every cold start --
# Building it costs ~16s here and 3-4x that on a small cloud instance, which is
# what pushed the first backtest past the reverse proxy's ~100s timeout (a 502
# on a perfectly healthy app). Ship it prebuilt instead.
#
# It MUST be built against the TRIMMED artifact and the REWRITTEN meta.json,
# because edge_lib fingerprints both. Build it against the full local artifact
# and every fingerprint check on the host misses -- shipping a 64MB file that is
# silently never used, with no error to tell you. So: swap the trimmed files in,
# build, swap back. try/finally, because leaving the developer's full artifact
# replaced by the deploy-trimmed one would be a nasty parting gift.
_meta_bytes = None
if _trimmed is not None:
    _mp = config.ARTIFACT_DIR / "meta.json"
    _mj = _json.loads(_mp.read_text(encoding="utf-8"))
    _mj["n_names"] = _mj["universe_size"] = len(_keep)
    _mj["deploy_trim_mcap"] = DEPLOY_MCAP_KEEP
    _meta_bytes = _json.dumps(_mj).encode()

_panels: list[pathlib.Path] = []
if os.environ.get("SKIP_PANEL_PRECOMPUTE") != "1":
    _live_art = config.ARTIFACT_DIR / "backtest_data.pkl"
    _mp = config.ARTIFACT_DIR / "meta.json"
    _bak_art = _bak_meta = None
    try:
        if _trimmed is not None:
            _bak_art = pathlib.Path(tempfile.mkstemp(suffix=".pkl")[1])
            shutil.copy(_live_art, _bak_art); shutil.copy(_trimmed, _live_art)
            _bak_meta = _mp.read_bytes(); _mp.write_bytes(_meta_bytes)

        # BUILD THE PANEL UNDER PRODUCTION'S UNIVERSE RULE, NOT THIS MACHINE'S.
        #
        # edge_lib decides USE_PIT_UNIVERSE at import from whatever pit_universe
        # auto-detects. This machine HAS a real PIT source, so it defaults True;
        # render.yaml pins EDGE_USE_PIT_UNIVERSE=0 so production runs the
        # top-N-by-market-cap proxy every backtest was validated against.
        #
        # _panel_fingerprint hashes that flag. A panel built here with PIT=True
        # therefore NEVER matches on the host -- the shipped panel was rejected on
        # arrival and every cold start rebuilt it from scratch, for months,
        # however many times the bundle was republished. It was also the wrong
        # universe: PIT keeps every member (~4,169 rows/rebalance) while the proxy
        # caps at 1,000, so the two are different models, not just different keys.
        #
        # Set BEFORE importing edge_lib: the flag is read at import time.
        os.environ["EDGE_USE_PIT_UNIVERSE"] = "0"
        import edge_lib as E
        if E.USE_PIT_UNIVERSE:                      # belt and braces if imported earlier
            E.USE_PIT_UNIVERSE = False
        E.reset_caches()
        print(f"      building panel with USE_PIT_UNIVERSE={E.USE_PIT_UNIVERSE} "
              f"(must match render.yaml's EDGE_USE_PIT_UNIVERSE=0)")
        for _pth in sorted(config.CACHE_DIR.glob("panel_h*.pkl")):
            _pth.unlink()          # never ship a panel from a previous artifact
        _t0 = time.time()
        # clock_spec, not a bare hold: since the grid became calendar-anchored the
        # panel is keyed on (hold, rebal_months) and the FILENAME carries both.
        # Precomputing with hold alone would build the legacy-stride panel and
        # ship it under a name the host never looks up -- a full cold rebuild on
        # every start, with a shipped panel sitting right next to it, and no error.
        E.load_edge_panel(universe=E.UNIVERSE, offset_days=0,
                          **E.clock_spec(E.EDGE_SPEC["hold"]))
        # Record WHAT THE PANEL WAS BUILT UNDER, so publish_bundle can refuse a
        # panel the host will reject. The fingerprint cannot be recomputed after
        # the fact -- it hashes the artifact size and meta.json, and both are
        # swapped back to the full research copies below -- so the knobs are
        # recorded instead. USE_PIT_UNIVERSE is the one that silently differed:
        # this machine auto-detects a PIT source, production pins it off.
        _clock = E.clock_spec(E.EDGE_SPEC["hold"])
        (config.ARTIFACT_DIR / "panel_build.json").write_text(_json.dumps({
            "use_pit_universe": bool(E.USE_PIT_UNIVERSE),
            "delist_haircut": E.DELIST_HAIRCUT,
            "hold": _clock["hold"], "rebal_months": _clock["rebal_months"],
            "universe": E.UNIVERSE, "lb": E.LB,
        }), encoding="utf-8")
        _panels = sorted(config.CACHE_DIR.glob("panel_h*.pkl"))
        print(f"precomputed panel: {', '.join(p.name for p in _panels)} "
              f"({sum(p.stat().st_size for p in _panels)/1e6:.0f}MB, {time.time()-_t0:.0f}s)")
    finally:
        if _bak_art is not None:
            shutil.copy(_bak_art, _live_art); _bak_art.unlink()
        if _bak_meta is not None:
            _mp.write_bytes(_bak_meta)

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    # the model artifacts (factor snapshot + the backtest panel pickle)
    for p in sorted(config.ARTIFACT_DIR.glob("*")):
        if not p.is_file() or _is_backup(p.name):
            continue
        if p.name == "backtest_data.pkl" and _trimmed is not None:
            z.write(_trimmed, p.relative_to(config.ROOT)); n += 1
            continue
        if p.name == "meta.json" and _meta_bytes is not None:
            # Rewrite the count to match what actually ships. Copying the
            # untrimmed meta made the live footer claim 12,164 names for a
            # bundle carrying 2,976 -- a number the deployed site displays.
            # Same bytes the panel was fingerprinted against, or the host misses.
            z.writestr(str(p.relative_to(config.ROOT)), _meta_bytes.decode()); n += 1
            continue
        z.write(p, p.relative_to(config.ROOT)); n += 1
    # cached yfinance benchmarks/gold (yf_*.pkl) so charts work without live calls
    for p in sorted(config.CACHE_DIR.glob("yf_*.pkl")):
        z.write(p, p.relative_to(config.ROOT)); n += 1
    # prebuilt panel(s): the whole point of the cold-start fix
    for p in _panels:
        z.write(p, p.relative_to(config.ROOT)); n += 1

if _trimmed is not None and _trimmed.exists():
    _trimmed.unlink()

size = OUT.stat().st_size / 1e6
print(f"Wrote {OUT} ({size:.0f} MB, {n} files, source={src}).")
print("Zip layout: data/artifacts/... and data/cache/yf_*.pkl (relative to project root).")
print("Next: upload it and set DATA_URL to its download link on Render.")
