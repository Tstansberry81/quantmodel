"""Fetch & assemble factor data per company.

Fundamentals come from fiscal.ai's ratios endpoint (one batched call per name,
all history). Price-derived factors (momentum 12-1, Sharpe, Sortino) are
computed from the daily price series. Everything is keyed so the same cached
data drives both the live screen and the historical backtest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from qmodel import fiscal
from qmodel.equations import all_ratio_ids


# ---- price-derived metrics -------------------------------------------------

def price_series(company_key: str) -> pd.Series:
    raw = fiscal.stock_prices(company_key)
    if not raw:
        return pd.Series(dtype=float)
    df = pd.DataFrame(raw)
    if "date" not in df or "price" not in df:
        return pd.Series(dtype=float)
    s = pd.Series(df["price"].values, index=pd.to_datetime(df["date"]))
    return s.sort_index().dropna()


def momentum_12_1(prices: pd.Series, asof: pd.Timestamp | None = None) -> float:
    """12-month return skipping the most recent month (classic 12-1)."""
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < 252:
        return np.nan
    p_now = prices.iloc[-21]          # ~1 month ago
    p_then = prices.iloc[-252]        # ~12 months ago
    if p_then <= 0:
        return np.nan
    return p_now / p_then - 1.0


def momentum_accel(prices: pd.Series, asof: pd.Timestamp | None = None) -> float:
    """Momentum acceleration: recent 3-month return minus the prior 3-month
    return. Positive = momentum is increasing (the 'change in momentum')."""
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < 127:
        return np.nan
    recent = prices.iloc[-1] / prices.iloc[-63] - 1.0
    prior = prices.iloc[-63] / prices.iloc[-126] - 1.0
    return float(recent - prior)


def momentum_accel(prices: pd.Series, asof: pd.Timestamp | None = None) -> float:
    """Acceleration = change in 6-month momentum (Ardila-Sornette 2020).

    recent 6m return minus the prior 6m return -> momentum that is *increasing*.
    """
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < 252:
        return np.nan
    p0, p6, p12 = prices.iloc[-1], prices.iloc[-126], prices.iloc[-252]
    if p6 <= 0 or p12 <= 0:
        return np.nan
    return float((p0 / p6 - 1.0) - (p6 / p12 - 1.0))


def residual_momentum(prices: pd.Series, market: pd.Series,
                      asof: pd.Timestamp | None = None,
                      lookback: int = 252, skip: int = 21) -> float:
    """Risk-adjusted residual (beta-neutral) 12-1 momentum (Blitz-Huij-Martens 2011).

    Cumulative residual return (from a market regression) over t-12..t-1 months,
    scaled by residual volatility. Strips out market-beta noise.
    """
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < lookback + 1 or market is None or len(market) == 0:
        return np.nan
    ret = prices.pct_change()
    mret = market.reindex(prices.index, method="ffill").pct_change()
    w, wm = ret.iloc[-lookback:], mret.iloc[-lookback:]
    mask = w.notna() & wm.notna()
    rr, mm = w[mask].values, wm[mask].values
    if len(mm) < 60 or mm.var() == 0:
        return np.nan
    beta = np.cov(rr, mm)[0, 1] / mm.var()
    resid = rr - (rr.mean() - beta * mm.mean()) - beta * mm
    rvol = resid.std(ddof=1)
    if rvol <= 0 or resid.size <= skip:
        return np.nan
    return float(resid[:-skip].sum() / rvol)


def mom_52w_high(prices: pd.Series, asof: pd.Timestamp | None = None) -> float:
    """Nearness to the 52-week high (George-Hwang 2004): price / 252-day max."""
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < 60:
        return np.nan
    window = prices.iloc[-252:]
    hi = window.max()
    return float(window.iloc[-1] / hi) if hi > 0 else np.nan


def _ann_sharpe_sortino(prices: pd.Series, lookback: int = 252,
                        asof: pd.Timestamp | None = None) -> tuple[float, float]:
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < lookback + 1:
        lookback = len(prices) - 1
    if lookback < 60:
        return np.nan, np.nan
    rets = prices.iloc[-lookback:].pct_change().dropna()
    if rets.std() == 0 or len(rets) < 30:
        return np.nan, np.nan
    rf_daily = config.RISK_FREE_ANNUAL / config.TRADING_DAYS
    excess = rets - rf_daily
    sharpe = np.sqrt(config.TRADING_DAYS) * excess.mean() / rets.std()
    downside = rets[rets < 0]
    dd = downside.std()
    sortino = (np.sqrt(config.TRADING_DAYS) * excess.mean() / dd) if dd and dd > 0 else np.nan
    return float(sharpe), float(sortino)


def realized_vol(prices: pd.Series, lookback: int = 252,
                 asof: pd.Timestamp | None = None) -> float:
    """Annualized realized volatility (std of daily returns x sqrt(252))."""
    if asof is not None:
        prices = prices[prices.index <= asof]
    if len(prices) < 60:
        return np.nan
    rets = prices.iloc[-lookback:].pct_change().dropna()
    if len(rets) < 30:
        return np.nan
    return float(rets.std() * np.sqrt(config.TRADING_DAYS))


def horizon_returns(prices: pd.Series) -> dict:
    """Trailing total return over each configured interval."""
    out = {}
    for label, ndays in config.INTERVALS.items():
        if len(prices) > ndays:
            out[label] = float(prices.iloc[-1] / prices.iloc[-1 - ndays] - 1.0)
        else:
            out[label] = None
    return out


# ---- fundamentals ----------------------------------------------------------

def _latest_metrics(ratio_json: dict) -> dict:
    """Most recent period's metric values from a ratios response."""
    if not ratio_json or "data" not in ratio_json or not ratio_json["data"]:
        return {}
    periods = sorted(ratio_json["data"], key=lambda d: d.get("reportDate", ""))
    return periods[-1].get("metricValues", {}) or {}


def fundamentals_history(ratio_json: dict) -> pd.DataFrame:
    """All historical periods as a DataFrame indexed by reportDate."""
    if not ratio_json or "data" not in ratio_json:
        return pd.DataFrame()
    rows = []
    for p in ratio_json["data"]:
        row = {"report_date": p.get("reportDate")}
        row.update(p.get("metricValues", {}) or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "report_date" in df:
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date").set_index("report_date")
    return df


def build_factor_row(member: dict, prices: pd.Series | None = None,
                     ratio_json: dict | None = None,
                     market: pd.Series | None = None) -> dict | None:
    """Assemble one universe member's full factor snapshot.

    prices: total-return (dividend-adjusted) daily series. If None, falls back
    to fiscal.ai split-adjusted prices.
    """
    ck = member["company_key"]
    rj = ratio_json if ratio_json is not None else fiscal.ratios(ck, all_ratio_ids(), period_type="annual")
    if prices is None or len(prices) == 0:
        prices = price_series(ck)
    if (rj is None or not rj.get("data")) and (prices is None or prices.empty):
        return None

    latest = _latest_metrics(rj or {})
    from qmodel.equations import FACTORS, EXTRA_RATIOS

    row = {
        "company_key": ck, "ticker": member["ticker"], "name": member["name"],
        "sector": member["sector"], "industry": member["industry"],
        "market_cap_rank": member.get("market_cap_rank"),
        "trading_status": member.get("trading_status", "Active"),
    }
    # fundamental factors
    for fname, f in FACTORS.items():
        if f["ratio_id"]:
            row[fname] = latest.get(f["ratio_id"], np.nan)
    for ename, rid in EXTRA_RATIOS.items():
        row[ename] = latest.get(rid, np.nan)

    # price factors (total return)
    row["momentum"] = momentum_12_1(prices)
    row["mom_accel"] = momentum_accel(prices)
    row["mom_52w_high"] = mom_52w_high(prices)
    row["reversal_1m"] = float(prices.iloc[-1] / prices.iloc[-21] - 1.0) if len(prices) >= 21 else np.nan
    row["resid_mom"] = residual_momentum(prices, market) if market is not None else np.nan
    sh, so = _ann_sharpe_sortino(prices)
    row["sharpe"], row["sortino"] = sh, so
    row["volatility"] = realized_vol(prices)
    row["last_price"] = float(prices.iloc[-1]) if len(prices) else np.nan
    row["returns"] = horizon_returns(prices)
    return row
