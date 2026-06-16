"""Market data for benchmarks + gold via yfinance (fiscal.ai has no index/ETF prices).

Cached to parquet so we don't re-download on every request.
"""
from __future__ import annotations
import time
from pathlib import Path

import pandas as pd

import config

_CACHE_TTL_HOURS = 12


def _cache_file(symbol: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("=", "_").replace(".", "_")
    return config.CACHE_DIR / f"yf_{safe}.pkl"


def get_prices(symbol: str, period: str = "max") -> pd.Series:
    """Return a daily close price Series indexed by date for a yfinance symbol."""
    cf = _cache_file(symbol)
    if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < _CACHE_TTL_HOURS:
        s = pd.read_pickle(cf)
        s.index = pd.to_datetime(s.index)
        return s

    import yfinance as yf
    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        if cf.exists():  # fall back to stale cache
            s = pd.read_pickle(cf); s.index = pd.to_datetime(s.index); return s
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):       # yfinance multiindex columns
        close = close.iloc[:, 0]
    out = pd.Series(close.values, index=pd.to_datetime(df.index), name="close")
    out.index.name = "date"
    out.to_pickle(cf)
    return out


def benchmarks() -> dict[str, pd.Series]:
    return {
        "SP500": get_prices(config.BENCH_SP500),
        "NASDAQ": get_prices(config.BENCH_NASDAQ),
        "GOLD": get_prices(config.GOLD_SYMBOL),
    }


def yahoo_symbol(ticker: str) -> str:
    """Map a fiscal.ai ticker to its Yahoo form (e.g. BRK.A -> BRK-A)."""
    return ticker.replace(".", "-")


def universe_total_return_prices(tickers: list[str], period: str = "max",
                                 chunk: int = 150) -> dict[str, pd.Series]:
    """Batch-download dividend+split-adjusted (total-return) daily closes.

    Returns {original_ticker: close Series}. Cached as one pickle keyed by the
    sorted ticker set so repeated builds are fast.
    """
    import hashlib
    key = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    cf = config.CACHE_DIR / f"yf_universe_{key}.pkl"
    if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < _CACHE_TTL_HOURS:
        return pd.read_pickle(cf)

    import yfinance as yf
    out: dict[str, pd.Series] = {}
    ymap = {yahoo_symbol(t): t for t in tickers}
    ysyms = list(ymap.keys())
    for i in range(0, len(ysyms), chunk):
        batch = ysyms[i:i + chunk]
        try:
            df = yf.download(batch, period=period, interval="1d", progress=False,
                             auto_adjust=True, group_by="ticker", threads=True)
        except Exception:
            continue
        for ys in batch:
            try:
                sub = df[ys]["Close"] if len(batch) > 1 else df["Close"]
                s = sub.dropna()
                if len(s):
                    s.index = pd.to_datetime(s.index)
                    out[ymap[ys]] = s
            except Exception:
                continue
    if out:
        pd.to_pickle(out, cf)
    return out
