"""The Edge -- shared harness for the SHORT-TERM TRADING model (Porter & Co).

This is the trading product, NOT the long-term fundamental engine. It scans the
Russell-1000-proxy universe and ranks on MARKET DATA ONLY (short-horizon momentum
/ relative strength / acceleration -- BIC-on-a-short-clock), no fundamentals.
Holding period ~30-60 days. Two proprietary overlays get tested on top:
  * correlation cap (<=0.50) so the basket isn't eight versions of one bet,
  * a 200-day-MA regime switch that raises cash below the line.

Original long-term model is preserved untouched in qmodel/ (and qmodel_original/).
This module reuses qmodel's cached data loaders + daily return matrix but builds
its own market-data-only panel and evaluates over 1Y / 2Y / 5Y windows.

No statsmodels in the venv; any stats are done by hand in numpy.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from functools import lru_cache
import numpy as np, pandas as pd
import os
import config
from qmodel import engine
import tech_bias_lib as TB           # reuse daily_return_matrix + benchmark loaders
import pit_universe as PIT           # pluggable PIT Russell-1000 universe layer

# Restrict each rebalance to REAL point-in-time Russell-1000 members (and price
# delisted names) when a real PIT source is present, instead of the survivorship-
# biased top-1000-by-mcap proxy. Defaults to whatever data is actually available:
# True iff pit_universe found a real source (Norgate or a user CSV). Override with
# the EDGE_USE_PIT_UNIVERSE env var ("1"/"0").
_env = os.environ.get("EDGE_USE_PIT_UNIVERSE")
USE_PIT_UNIVERSE = (_env == "1") if _env in ("0", "1") else PIT.is_real_pit()

HOLD = 21                            # ~30 calendar days (a 1-month trading clock)
LB = 251                             # trailing window for signals / beta
PPY = 252.0 / HOLD
RF_PER = config.RISK_FREE_ANNUAL / PPY
N = 10                               # basket size (agents may vary)
UNIVERSE = 1000                      # Russell-1000 proxy: top-N by point-in-time mcap

# short-horizon signal columns the trading model ranks on (market data only)
SIGNAL_COLS = ["ret_21", "ret_63", "ret_126", "accel", "hi_252", "rs_63"]


class EdgePanel:
    def __init__(self, panels, bdates, spxf, ndxf, ma200_on, mkt_daily, sectors, hold):
        self.panels = panels          # list[DataFrame] per rebalance
        self.bdates = bdates          # DatetimeIndex
        self.spxf = spxf              # S&P fwd return per period (over the FULL hold window)
        self.ndxf = ndxf              # Nasdaq fwd return per period (over the FULL hold window)
        self.ma200_on = ma200_on      # bool array: market above its 200-day MA at d
        self.mkt_daily = mkt_daily    # S&P daily returns (Series)
        self.sectors = sectors
        self.T = len(panels)
        self.hold = hold                          # holding period in trading days
        self.ppy = 252.0 / hold                   # rebalances per year (annualization)
        self.rf_per = config.RISK_FREE_ANNUAL / self.ppy   # per-period risk-free


@lru_cache(maxsize=6)
def load_edge_panel(hold: int = HOLD, universe: int = UNIVERSE) -> EdgePanel:
    """Build the trading panel once. Each row = one candidate's short-horizon
    market-data signals + forward return. NO fundamentals."""
    data = engine._load_bt_data(); bm = engine._benchmarks_cached()
    spx = bm["SP500"].dropna(); mret_full = spx.pct_change()
    ndx = bm["NASDAQ"].dropna()
    ma200 = spx.rolling(200).mean()

    prep = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB + hold + 2:
            continue
        arr = np.asarray(pr.values, float)
        pidx = np.asarray(pr.index.values, "datetime64[ns]")
        rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:] / arr[:-1] - 1
        mret = np.asarray(mret_full.reindex(pr.index, method="ffill").values, float)
        fh = blob.get("fund_hist"); fidx = None; mcap = None; rgr = None
        if fh is not None and not fh.empty:
            fidx = np.asarray(fh.index.values, "datetime64[ns]")
            if "calculated_market_cap" in fh.columns:
                mcap = np.asarray(fh["calculated_market_cap"].values, float)
            if "growth_revenue_1y" in fh.columns:
                rgr = np.asarray(fh["growth_revenue_1y"].values, float)
        prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret,
                    "fidx": fidx, "mcap": mcap, "rgr": rgr,
                    "sector": blob.get("meta", {}).get("sector", "Unknown")}

    def asof(fidx, varr, dlag):
        if fidx is None or varr is None:
            return np.nan
        p = int(np.searchsorted(fidx, dlag, side="right")) - 1
        return varr[p] if p >= 0 else np.nan

    lag = np.timedelta64(90, "D")
    monthly = spx.index[spx.index >= spx.index.min() + pd.Timedelta(days=400)]
    rebal = list(monthly[:-hold][::hold])
    panels, bdates, spxf, ndxf, ma200_on = [], [], [], [], []
    use_pit = USE_PIT_UNIVERSE and PIT.is_real_pit()

    for d in rebal:
        dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt - lag
        # market trailing returns for relative strength
        mpos = int(np.searchsorted(spx.index.values.astype("datetime64[ns]"), dt, side="right")) - 1
        m63 = float(spx.iloc[mpos] / spx.iloc[mpos - 63] - 1) if mpos >= 63 else 0.0
        # REAL PIT universe: restrict to actual Russell-1000 members on date d
        # (includes delisted names that were members then). Empty set => the
        # adapter can't answer this date, so fall back to the mcap proxy below.
        pit_set = PIT.pit_members(pd.Timestamp(d)) if use_pit else frozenset()
        cand = []
        for ck, P in prep.items():
            if pit_set and ck not in pit_set:
                continue
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB or (len(P["arr"]) - 1 - pos) < hold:
                continue
            mc = asof(P["fidx"], P["mcap"], dlag)
            if mc is None or np.isnan(mc):
                continue
            cand.append((mc, ck, pos))
        if len(cand) < 50:
            continue
        cand.sort(key=lambda x: x[0], reverse=True)
        # With real PIT membership we keep ALL members (membership IS the
        # universe definition); only the proxy needs the top-N-by-mcap cut.
        if not pit_set:
            cand = cand[:universe]
        rows = []
        for mc, ck, pos in cand:
            P = prep[ck]; arr = P["arr"]; w = P["rets"][pos - LB + 1: pos + 1]
            wm = P["mret"][pos - LB + 1: pos + 1]
            sd = w.std(ddof=1)
            ok = ~np.isnan(w) & ~np.isnan(wm); beta = np.nan
            if ok.sum() > 60 and wm[ok].var() > 0:
                beta = float(np.cov(w[ok], wm[ok])[0, 1] / wm[ok].var())
            ret_63 = arr[pos] / arr[pos - 63] - 1
            rows.append({
                "company_key": ck, "sector": P["sector"], "pit_mcap": mc, "beta": beta,
                "ret_21": arr[pos] / arr[pos - 21] - 1,
                "ret_63": ret_63,
                "ret_126": arr[pos] / arr[pos - 126] - 1,
                "accel": ret_63 - (arr[pos - 63] / arr[pos - 126] - 1),
                "hi_252": arr[pos] / arr[pos - 251: pos + 1].max(),
                "rs_63": ret_63 - m63,                  # relative strength vs market
                "vol_21": float(w[-21:].std(ddof=1) * np.sqrt(252)) if sd > 0 else np.nan,
                "rev_growth": asof(P["fidx"], P["rgr"], dlag),   # YoY rev growth (90d-lagged)
                "fwd_ret": arr[pos + hold] / arr[pos] - 1,
            })
        panels.append(pd.DataFrame(rows)); bdates.append(pd.Timestamp(d))
        # benchmark forward return over the FULL hold-day window (hold-aware!)
        spxf.append(_bench_fwd_hold(spx, d, hold))
        ndxf.append(_bench_fwd_hold(ndx, d, hold))
        ma200_on.append(bool(spx.iloc[mpos] > ma200.iloc[mpos]) if not np.isnan(ma200.iloc[mpos]) else True)

    sectors = sorted({s for df in panels for s in df["sector"].unique() if str(s).lower() != "unknown"})
    return EdgePanel(panels, pd.DatetimeIndex(bdates), np.array(spxf), np.array(ndxf),
                     np.array(ma200_on, bool), mret_full, sectors, hold)


# ---- ranking + selection ----------------------------------------------------
def _z(s):
    s = pd.to_numeric(s, errors="coerce"); mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else pd.Series(0.0, index=s.index)


def score(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Cross-sectional composite of short-horizon signals (BIC-on-short-clock).
    Default = pure 3-month momentum / relative strength."""
    weights = weights or {"ret_63": 1.0}
    comp = pd.Series(0.0, index=df.index)
    for col, wt in weights.items():
        if col in df.columns:
            comp = comp + wt * _z(df[col]).fillna(0.0)
    return comp


def select_topN(df: pd.DataFrame, n: int = N, weights: dict | None = None) -> list:
    d = df.dropna(subset=["fwd_ret"]).copy()
    d["_c"] = score(d, weights)
    return list(d.sort_values("_c", ascending=False).head(n)["company_key"])


def corr_cap_select(df, asof=None, n=N, weights=None, cap=0.50, lookback=126):
    """Greedy basket: walk names best-first, admit a name only if its return
    correlation with every already-admitted name is <= cap.

    `asof` (the rebalance date) is REQUIRED for a point-in-time window -- the
    correlation is judged on the `lookback` trading days ending at `asof`, never
    future data. (Omitting it falls back to the most-recent window, which is only
    valid for a live pick, not a historical backtest.)"""
    d = df.dropna(subset=["fwd_ret"]).copy()
    d["_c"] = score(d, weights)
    order = list(d.sort_values("_c", ascending=False)["company_key"])
    M = TB.daily_return_matrix()
    win = (M[M.index <= asof] if asof is not None else M).tail(lookback)
    chosen = []
    for ck in order:
        if ck not in win.columns:
            continue
        if not chosen:
            chosen.append(ck); continue
        c = win[[ck] + chosen].corr().iloc[0, 1:].abs().max()
        if pd.isna(c) or c <= cap:
            chosen.append(ck)
        if len(chosen) >= n:
            break
    if len(chosen) < n:                       # backfill if the cap was too tight
        for ck in order:
            if ck not in chosen:
                chosen.append(ck)
            if len(chosen) >= n:
                break
    return chosen[:n]


# ---- returns + evaluation ----------------------------------------------------
def daily_from_holdings(pan: EdgePanel, holds: list, hold: int = HOLD) -> pd.Series:
    M = TB.daily_return_matrix(); cal = M.index
    pieces = []
    for i, d in enumerate(pan.bdates):
        cks = [c for c in holds[i] if c in M.columns]
        loc = int(cal.searchsorted(d, side="right"))
        win = cal[loc: loc + hold]
        if len(win) == 0 or not cks:
            continue
        pieces.append(M.loc[win, cks].mean(axis=1))
    if not pieces:
        return pd.Series(dtype=float)
    s = pd.concat(pieces)
    return s[~s.index.duplicated(keep="first")].fillna(0.0)


def period_returns(pan: EdgePanel, holds: list, regime_cash: bool = False) -> np.ndarray:
    """Equal-weight top-N forward returns per period. If regime_cash, periods
    where the market is below its 200-day MA earn cash instead."""
    out = []
    for i, df in enumerate(pan.panels):
        sel = df[df["company_key"].isin(holds[i])]
        r = float(sel["fwd_ret"].mean()) if len(sel) else 0.0
        if regime_cash and not pan.ma200_on[i]:
            r = pan.rf_per
        out.append(r)
    return np.array(out)


def perf(rets, ppy=PPY):
    r = np.nan_to_num(np.asarray(rets, float))
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (ppy / len(r)) - 1 if len(r) and eq[-1] > 0 else -1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(r) else 0.0
    sh = float(np.sqrt(ppy) * np.nanmean(r) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
    return float(cagr), dd, sh


def eval_windows(pan: EdgePanel, rets, label=""):
    """Print CAGR / Sharpe / maxDD over trailing 1Y / 2Y / 5Y / full, vs S&P,
    using the panel's true rebalances-per-year for annualization."""
    rets = np.asarray(rets, float)
    ppy = pan.ppy
    windows = (("1Y", int(round(ppy))), ("2Y", int(round(2 * ppy))),
               ("5Y", int(round(5 * ppy))), ("MAX", None))
    print(f"  {label}")
    print(f"    {'window':<6}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}   {'vs S&P CAGR':>12}")
    for wname, k in windows:
        kk = len(rets) if k is None else min(k, len(rets))
        c, dd, sh = perf(rets[-kk:], ppy)
        cs, _, _ = perf(pan.spxf[-kk:], ppy)
        print(f"    {wname:<6}{c*100:7.1f}%{sh:8.2f}{dd*100:7.0f}%   {(c-cs)*100:>+11.1f}%")
    return rets


def _bench_fwd_hold(series, d, hold):
    past = series[series.index <= d]; fwd = series[series.index > d]
    if past.empty or len(fwd) < 1:
        return np.nan
    return float(fwd.iloc[min(hold, len(fwd)) - 1] / past.iloc[-1] - 1)


# ---- the robust, tradeable Edge: signal + liquidity + corr-cap + regime + costs
EDGE_SPEC = dict(signal={"accel": 1.0}, hold=42, n=20, mcap_floor=2e9,
                 corr_cap=0.50, corr_lookback=126, regime_expo=0.25, cost_bps=10.0,
                 growth_mix=0.75,          # fraction of the basket drawn from the >=15%-rev-growth pool
                 growth_thresh=0.15)       # YoY revenue-growth bar that defines a "growth" name


def _blend_select(d, asof, n, weights, corr_cap, corr_lookback, growth_mix, growth_thresh):
    """Build the basket as K growth-gated picks + (n-K) pure-signal picks, where
    K = round(growth_mix * n). growth_mix=0 -> pure signal; 1 -> all-growth."""
    def pick(frame, k):
        if k <= 0 or len(frame) < 1:
            return []
        return (corr_cap_select(frame, asof=asof, n=k, weights=weights, cap=corr_cap,
                                lookback=corr_lookback) if corr_cap
                else select_topN(frame, n=k, weights=weights))
    K = int(round(growth_mix * n))
    gpick = []
    if K > 0:
        g = d[pd.to_numeric(d["rev_growth"], errors="coerce") >= growth_thresh]
        gpick = pick(g, K)
    sel = list(gpick)
    for ck in pick(d, n + len(gpick)):          # fill remainder from the full pool
        if len(sel) >= n:
            break
        if ck not in sel:
            sel.append(ck)
    return sel[:n]


@lru_cache(maxsize=16)
def _edge_full(hold, n, mcap_floor, corr_cap, corr_lookback, regime_expo, cost_bps,
               signal_key, growth_mix=0.0, growth_thresh=0.15):
    """Compute the full-history Edge once (cached). Returns the per-period gross
    & net returns, turnover, and aligned S&P / Nasdaq returns.

    growth_mix in [0,1] sets how much of the basket must come from names with YoY
    revenue growth >= growth_thresh (a fundamentals tilt; 0 = pure market-data)."""
    weights = dict(signal_key)
    pan = load_edge_panel(hold=hold)
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= mcap_floor] if mcap_floor else df
        cks = _blend_select(d, pan.bdates[i], n, weights, corr_cap, corr_lookback,
                            growth_mix, growth_thresh)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    gross = period_returns(pan, holds)
    if regime_expo is not None:
        gross = np.where(pan.ma200_on, gross, regime_expo * gross + (1 - regime_expo) * pan.rf_per)
    turn = np.array(turn)
    net = gross - (cost_bps / 1e4) * turn
    return pan.bdates, gross, net, turn, pan.spxf, pan.ndxf


def run_edge_backtest(window: str = "MAX", spec: dict | None = None) -> dict:
    """Backtest the tradeable Edge over a trailing window. Returns curves +
    performance in the same shape the website's chart code expects (NET of costs)."""
    s = {**EDGE_SPEC, **(spec or {})}
    pan = load_edge_panel(hold=s["hold"])
    ppy = pan.ppy                                      # true rebalances/year for this hold
    bdates, gross, net, turn, spxf, ndxf = _edge_full(
        s["hold"], s["n"], s["mcap_floor"], s["corr_cap"], s["corr_lookback"],
        s["regime_expo"], s["cost_bps"], tuple(sorted(s["signal"].items())),
        float(s.get("growth_mix", 0.0) or 0.0), float(s.get("growth_thresh", 0.15)))
    T = len(net)
    wmap = {"1Y": int(round(ppy)), "2Y": int(round(2 * ppy)),
            "3Y": int(round(3 * ppy)), "5Y": int(round(5 * ppy)), "MAX": T}
    k = min(wmap.get(window, T), T)
    if k < 2:
        return {"ok": False, "reason": "window too short"}
    sl = slice(T - k, T)
    mr, sr, nr, dts = net[sl], spxf[sl], ndxf[sl], bdates[T - k:T]

    def curve(r):
        return [round(float(x), 4) for x in np.cumprod(1 + np.nan_to_num(r))]

    def stats(r):
        c, dd, sh = perf(r, ppy)
        tot = float(np.cumprod(1 + np.nan_to_num(r))[-1] - 1)
        return {"cagr": c, "sharpe": sh, "max_drawdown": dd, "total_return": tot}

    windows = []
    for wn in ("1Y", "2Y", "5Y", "MAX"):
        kk = min(wmap[wn], T)
        cg, dd, sh = perf(net[T - kk:T], ppy); cs, _, ss = perf(spxf[T - kk:T], ppy)
        windows.append({"w": wn, "cagr": cg, "sharpe": sh, "dd": dd,
                        "sp_cagr": cs, "excess": cg - cs})
    avg_to = float(np.mean(turn[sl]))
    gc = perf(gross[sl], ppy)[0]
    return {
        "ok": True, "n_rebalances": k,
        "period": [str(dts[0].date()), str(dts[-1].date())],
        "spec": {"signal": "acceleration (3m−prior3m)", "hold_days": s["hold"],
                 "n": s["n"], "mcap_floor_bn": s["mcap_floor"] / 1e9,
                 "corr_cap": s["corr_cap"], "regime_expo": s["regime_expo"],
                 "cost_bps": s["cost_bps"],
                 "growth_mix": float(s.get("growth_mix", 0.0) or 0.0),
                 "growth_thresh": float(s.get("growth_thresh", 0.15))},
        "performance": {"model": stats(mr), "sp500": stats(sr), "nasdaq": stats(nr)},
        "curves": {"dates": [str(d.date()) for d in dts],
                   "model": curve(mr), "sp500": curve(sr), "nasdaq": curve(nr)},
        "windows": windows,
        "turnover": {"per_rebalance": round(avg_to, 3), "annualized": round(avg_to * ppy, 1),
                     "cost_bps": s["cost_bps"], "gross_cagr": gc,
                     "net_cagr": stats(mr)["cagr"]},
    }


if __name__ == "__main__":
    print("Building Edge trading panel (one-time, ~1-2 min) ...")
    pan = load_edge_panel()
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}, "
          f"avg {np.mean([len(p) for p in pan.panels]):.0f} names; "
          f"market above 200dMA {pan.ma200_on.mean()*100:.0f}% of the time\n")
    holds = [select_topN(df) for df in pan.panels]
    base = period_returns(pan, holds)
    eval_windows(pan, base, "BASELINE: 3-month momentum, equal-weight top-10, no overlays")
