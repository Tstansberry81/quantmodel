"""The Edge -- self-contained data layer.

Everything the Edge needs to read its cached data, with NO dependency on the old
long-term ("Slow Burn") qmodel signal code. Provides:
  * load_bt_data()        -- per-name price + fundamentals history (backtest pickle)
  * benchmarks()          -- S&P / Nasdaq / gold daily closes (yfinance, cached)
  * meta()                -- artifact build metadata (for the site footer)
  * daily_return_matrix() -- daily returns for every name on the S&P calendar

Only depends on config + pandas/numpy + the on-disk caches under data/. The Edge
model (edge_lib), the tracker, the export, and the PIT layer all import THIS,
so the qmodel package and the tech-bias research modules can be removed.
"""
from __future__ import annotations
import json
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config

_CACHE_TTL_HOURS = 12
LB = 251                                  # min price history to include a name


# ---- artifacts --------------------------------------------------------------
# Which artifact file to read. Lets a research run point at a DIFFERENT vendor's
# panel (e.g. backtest_data.fiscal.pkl) without moving files around, so a
# same-model / different-data A/B is repeatable and can't leave the production
# artifact swapped out if it's interrupted. Unset = the live artifact.
ARTIFACT_FILE = os.environ.get("EDGE_ARTIFACT", "backtest_data.pkl")


@lru_cache(maxsize=1)
def load_bt_data() -> dict:
    """Per-name {prices, fund_hist, meta} from the backtest artifact pickle."""
    p = config.ARTIFACT_DIR / ARTIFACT_FILE
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def meta() -> dict:
    p = config.ARTIFACT_DIR / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def yahoo_symbol(ticker: str) -> str:
    """Map a fiscal.ai ticker to its Yahoo form (e.g. BRK.A -> BRK-A)."""
    return ticker.replace(".", "-")


# ---- benchmarks (yfinance, cached to data/cache/yf_*.pkl) -------------------
def _cache_file(symbol: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("=", "_").replace(".", "_")
    return config.CACHE_DIR / f"yf_{safe}.pkl"


def get_prices(symbol: str, period: str = "max") -> pd.Series:
    """Daily close price Series for a yfinance symbol (served from cache; only
    re-downloads when the cache is stale)."""
    cf = _cache_file(symbol)
    if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < _CACHE_TTL_HOURS:
        s = pd.read_pickle(cf); s.index = pd.to_datetime(s.index); return s
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval="1d",
                         progress=False, auto_adjust=True)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        if cf.exists():
            s = pd.read_pickle(cf); s.index = pd.to_datetime(s.index); return s
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.Series(close.values, index=pd.to_datetime(df.index), name="close")
    out.index.name = "date"
    out.to_pickle(cf)
    return out


@lru_cache(maxsize=1)
def benchmarks() -> dict:
    """S&P 500 TR / Nasdaq-100 TR / gold daily closes (cached)."""
    return {
        "SP500": get_prices(config.BENCH_SP500),
        "NASDAQ": get_prices(config.BENCH_NASDAQ),
        "GOLD": get_prices(config.GOLD_SYMBOL),
    }


# ---- daily return matrix (for the correlation cap) --------------------------
@lru_cache(maxsize=1)
def daily_return_matrix() -> pd.DataFrame:
    """Daily returns for every name, reindexed to the S&P trading calendar
    (forward-filled prices). Columns = company_key. Used by the correlation cap."""
    data = load_bt_data(); bm = benchmarks()
    cal = bm["SP500"].dropna().index
    cols = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB:
            continue
        cols[ck] = pr.reindex(cal, method="ffill").pct_change().values
    return pd.DataFrame(cols, index=cal)


# Backward-compatible aliases so research scripts that used the old qmodel.engine
# can `import edge_data as engine` as a drop-in replacement.
_load_bt_data = load_bt_data
_benchmarks_cached = benchmarks
