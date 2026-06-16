"""Glue layer the Flask app calls: load artifacts, score, detect regime,
build the portfolio, and run the backtest with a given set of params.
"""
from __future__ import annotations
import json
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from qmodel import scoring, portfolio, hmm_regime, backtest, market
from qmodel.equations import merge_overrides, FACTORS, EXTRA_RATIOS, SETTINGS, EQUATION_META


# ---- artifact loading (cached in memory) -----------------------------------

@lru_cache(maxsize=1)
def _load_factor_rows() -> tuple:
    p = config.ARTIFACT_DIR / "factors.json"
    if not p.exists():
        return tuple()
    return tuple(json.loads(p.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _load_bt_data():
    p = config.ARTIFACT_DIR / "backtest_data.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _benchmarks_cached():
    return market.benchmarks()


def meta() -> dict:
    p = config.ARTIFACT_DIR / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def factor_dataframe() -> pd.DataFrame:
    rows = _load_factor_rows()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(list(rows))
    return df


# ---- regime ----------------------------------------------------------------

def current_regime(params: dict) -> dict:
    bm = _benchmarks_cached()
    return hmm_regime.fit_regime(bm["SP500"], n_states=params["settings"]["hmm_states"])


# ---- main compute ----------------------------------------------------------

def compute_portfolio(overrides: dict | None = None) -> dict:
    params = merge_overrides(overrides)
    df = factor_dataframe()
    if df.empty:
        return {"ok": False, "reason": "No data artifacts. Run build_data.py first."}

    # Live portfolio can only hold currently-tradeable names; inactive/delisted
    # names exist only to de-bias the backtest.
    n_inactive = 0
    if "trading_status" in df.columns:
        n_inactive = int((df["trading_status"] != "Active").sum())
        df = df[df["trading_status"] == "Active"].copy()

    scored = scoring.compute_scores(df, params)
    reg = current_regime(params)
    # price series needed for Stage-3 covariance (non-equal scheme) or the
    # Stage-6 crash overlay (vol-targeting needs the book's recent volatility).
    price_data = None
    if (params["settings"].get("weight_scheme", "equal") != "equal"
            or params["settings"].get("vol_target")):
        price_data = {ck: b.get("prices") for ck, b in _load_bt_data().items()}
    sel = portfolio.select(scored, params, reg.get("current", "neutral"), price_data=price_data)

    # benchmark trailing returns per interval (for the "vs S&P 500" column)
    spx = _benchmarks_cached()["SP500"].dropna()
    bench_returns = {}
    for label, ndays in config.INTERVALS.items():
        bench_returns[label] = (float(spx.iloc[-1] / spx.iloc[-1 - ndays] - 1.0)
                                if len(spx) > ndays else None)

    # full ranked table (all names) for the portfolio page
    ranked = scoring.screen_flags(scored, params).sort_values("composite", ascending=False)
    table = []
    for _, r in ranked.head(200).iterrows():   # cap display for a snappy UI
        table.append({
            "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
            "company_key": r["company_key"],
            "composite": round(float(r["composite"]), 3),
            "screens_passed": int(r["screens_passed"]),
            "screens_total": int(r["screens_total"]),
            "returns": r.get("returns", {}),
            "metrics": {k: portfolio._num(r.get(k)) for k in
                        ("roic", "eps", "pe", "fcf_margin", "rev_growth",
                         "momentum", "sharpe", "sortino", "volatility")},
        })

    return {
        "ok": True,
        "portfolio": sel["holdings"],
        "regime": {"current": reg.get("current"), "ok": reg.get("ok"),
                   "since": reg.get("current_since"), "stats": reg.get("state_stats", {})},
        "gold_weight": sel["gold_weight"],
        "cash_weight": sel.get("cash_weight", 0.0),
        "vol_exposure": sel.get("vol_exposure", 1.0),
        "n_eligible": sel["n_eligible"],
        "ranked": table,
        "bench_returns": bench_returns,
        "meta": meta(),
    }


# Backtest lookback windows -> pandas DateOffset
_BT_WINDOWS = {
    "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1),
    "2Y": pd.DateOffset(years=2), "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10), "20Y": pd.DateOffset(years=20), "MAX": None,
}

# Holding period / rebalance interval -> trading days
_HOLD_DAYS = {"1W": 5, "2W": 10, "1M": 21, "3M": 63, "6M": 126, "12M": 252}


def compute_backtest(overrides: dict | None = None, window: str = "MAX",
                     hold: str = "1M") -> dict:
    params = merge_overrides(overrides)
    data = _load_bt_data()
    if not data:
        return {"ok": False, "reason": "No backtest data. Run build_data.py first."}
    bm = _benchmarks_cached()

    start = None
    window = (window or "MAX").upper()
    offset = _BT_WINDOWS.get(window)
    if offset is not None:
        end = bm["SP500"].dropna().index.max()
        start = (end - offset).strftime("%Y-%m-%d")
    hold_days = _HOLD_DAYS.get((hold or "1M").upper(), 21)
    res = backtest.run_backtest(data, bm, params, start=start, hold_days=hold_days)
    if isinstance(res, dict):
        res["window"] = window
        res["hold"] = (hold or "1M").upper()
    return res


# ---- per-stock vs S&P 500 (for the selector chart) -------------------------

def stock_vs_sp500(company_key: str) -> dict:
    data = _load_bt_data()
    bm = _benchmarks_cached()
    blob = data.get(company_key)
    if not blob or blob.get("prices") is None or len(blob["prices"]) == 0:
        return {"ok": False, "reason": "no price data for that stock"}
    s = blob["prices"].dropna()
    spx = bm["SP500"].dropna()
    # align on common dates, rebase to 100
    idx = s.index.intersection(spx.index)
    if len(idx) < 30:
        return {"ok": False, "reason": "insufficient overlap"}
    s, spx = s.reindex(idx), spx.reindex(idx)

    out_curves = {}
    for label, ndays in config.INTERVALS.items():
        if len(idx) <= ndays:
            continue
        ss, bb = s.iloc[-ndays:], spx.iloc[-ndays:]
        out_curves[label] = {
            "dates": [str(d.date()) for d in ss.index],
            "stock": [round(float(x / ss.iloc[0] * 100), 2) for x in ss],
            "sp500": [round(float(x / bb.iloc[0] * 100), 2) for x in bb],
            "stock_return": round(float(ss.iloc[-1] / ss.iloc[0] - 1), 4),
            "sp500_return": round(float(bb.iloc[-1] / bb.iloc[0] - 1), 4),
        }
    return {"ok": True, "company_key": company_key,
            "ticker": blob.get("meta", {}).get("ticker"),
            "name": blob.get("meta", {}).get("name"),
            "curves": out_curves}


def equations_schema() -> dict:
    """Describe every adjustable knob for the equation-editor UI."""
    factors = {}
    for k, v in FACTORS.items():
        m = EQUATION_META.get(k, {})
        factors[k] = {
            "label": v["label"], "threshold": v["threshold"], "op": v["op"],
            "weight": v["weight"], "unit": v["unit"], "desc": v["desc"],
            "higher_better": v["higher_better"],
            "symbol": m.get("symbol", v["label"]), "formula": m.get("formula", ""),
            "explain": m.get("explain", v["desc"]),
            "display_scale": m.get("display_scale", 1),
            "display_unit": m.get("display_unit", ""),
        }
    return {
        "factors": factors,
        "settings": {"winsor_pct": SETTINGS["winsor_pct"],
                     "sector_neutral": SETTINGS["sector_neutral"]},
    }
