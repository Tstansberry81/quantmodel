"""The Edge -- shared harness for the SHORT-TERM TRADING model (Porter & Co).

It scans the Russell-1000-proxy universe and ranks on MARKET DATA ONLY
(short-horizon momentum / relative strength / acceleration), no fundamentals apart
from an optional revenue-growth tilt. Holding period ~30-60 days. Two proprietary
overlays get tested on top:
  * correlation cap (<=0.50) so the basket isn't eight versions of one bet,
  * a 200-day-MA regime switch that raises cash below the line.

All data is read through the self-contained edge_data layer (cached prices,
benchmarks, daily-return matrix). Signals are formed on close t but the trade
enters at close t+1 (a realistic one-day execution lag). Evaluated over
1Y / 2Y / 5Y / MAX windows.

No statsmodels in the venv; any stats are done by hand in numpy.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from functools import lru_cache
import numpy as np, pandas as pd
import os
import config
import edge_data as D                # self-contained Edge data layer (no Slow Burn deps)
import pit_universe as PIT           # pluggable PIT Russell-1000 universe layer

# Restrict each rebalance to REAL point-in-time Russell-1000 members (and price
# delisted names) when a real PIT source is present, instead of the survivorship-
# biased top-1000-by-mcap proxy. Defaults to whatever data is actually available:
# True iff pit_universe found a real source (Norgate or a user CSV). Override with
# the EDGE_USE_PIT_UNIVERSE env var ("1"/"0").
_env = os.environ.get("EDGE_USE_PIT_UNIVERSE")
USE_PIT_UNIVERSE = (_env == "1") if _env in ("0", "1") else PIT.is_real_pit()

# Survivorship stress mode. By default a name whose price series ends inside
# the forward hold window is EXCLUDED from that rebalance's panel -- the
# backtest never holds a stock through its delisting, which flatters returns.
# Set EDGE_DELIST_HAIRCUT (e.g. "-0.30") to instead INCLUDE such names with
# fwd_ret truncated at their last trade and the haircut applied to the final
# price (a proxy for the delisting/OTC fade the data can't see). 0.0 = truncate
# only. This is a research/stress knob; leave unset for product behaviour.
_dh = os.environ.get("EDGE_DELIST_HAIRCUT")
try:
    DELIST_HAIRCUT: float | None = float(_dh) if _dh not in (None, "") else None
except ValueError:
    DELIST_HAIRCUT = None

HOLD = 21                            # ~30 calendar days (a 1-month trading clock)
LB = 251                             # trailing window for signals / beta
PPY = 252.0 / HOLD
RF_PER = config.RISK_FREE_ANNUAL / PPY
N = 10                               # basket size (agents may vary)
UNIVERSE = 1000                      # Russell-1000 proxy: top-N by point-in-time mcap

# short-horizon signal columns the trading model ranks on (market data only)
SIGNAL_COLS = ["ret_21", "ret_63", "ret_126", "ret_12_1", "accel", "hi_252", "rs_63"]

# The single source of truth for backtest/tracker window labels. app.py,
# edge_tracker_lib, export_vision, and the templates' window buttons must all
# agree with this list; everything but "MAX" is "<years>Y".
WINDOWS = ("1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "MAX")
# windows shown in the per-window summary table (3Y is selector-only)
SUMMARY_WINDOWS = tuple(w for w in WINDOWS if w != "3Y")


def window_k(window: str, n_days: int) -> int:
    """Trading days covered by a window label, capped at the series length.
    MAX (or anything unrecognised) = the whole series. The backtest measures on
    the DAILY curve, so windows are counted in trading days here; the tracker's
    per-rebalance log does its own rebalance-count math off the same WINDOWS."""
    if window == "MAX" or window not in WINDOWS:
        return n_days
    return min(int(window[:-1]) * 252, n_days)


class EdgePanel:
    def __init__(self, panels, bdates, spxf, ndxf, ma200_on, mkt_daily, sectors, hold,
                 live_panel=None, live_date=None, live_regime_on=True,
                 live_spx_todate=float("nan")):
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
        # ---- the still-open position (see load_edge_panel) --------------------
        # Deliberately kept OUT of `panels`: these rows have no realized forward
        # return, so the backtest must never see them. `fwd_todate` marks each
        # name to the latest close instead.
        self.live_panel = live_panel              # DataFrame | None
        self.live_date = live_date                # Timestamp | None
        self.live_regime_on = live_regime_on      # market above its 200dMA at live_date
        self.live_spx_todate = live_spx_todate    # S&P return since live_date


@lru_cache(maxsize=10)
def load_edge_panel(hold: int = HOLD, universe: int = UNIVERSE, offset_days: int = 0) -> EdgePanel:
    """Build the trading panel once. Each row = one candidate's short-horizon
    market-data signals + forward return. NO fundamentals."""
    data = D.load_bt_data(); bm = D.benchmarks()
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
    # offset_days shifts the whole rebalance grid (for staggered sleeves); 0 = default.
    monthly = monthly[offset_days:]
    rebal = list(monthly[:-hold][::hold])
    # The scheduled rebalance grid runs `hold` days past the backtest panel's
    # last date. That final entry is the position a live trader following the
    # Edge opened and is STILL HOLDING -- no realized forward return yet, which
    # is exactly why the backtest grid stops short of it. Build it here (same
    # prep, same rules) but keep it out of `panels`.
    sched = list(monthly[::hold])
    live_d = sched[-1] if (sched and (not rebal or sched[-1] != rebal[-1])) else None
    panels, bdates, spxf, ndxf, ma200_on = [], [], [], [], []
    live_rows, live_date, live_regime_on, live_spx_todate = None, None, True, np.nan
    use_pit = USE_PIT_UNIVERSE and PIT.is_real_pit()
    spx_end = np.datetime64(pd.Timestamp(spx.index[-1]), "ns")   # sample end, for delist detection

    for d in (rebal + ([live_d] if live_d is not None else [])):
        is_live = live_d is not None and d == live_d
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
            if pos < LB:
                continue
            n_fwd = len(P["arr"]) - 1 - pos
            full_fwd = n_fwd >= hold + 1          # +1: enter next day (t+1)
            if is_live:
                # No forward window exists yet for the open position. Require the
                # name to still be trading at the rebalance (so a ticker that died
                # months ago can't be resurrected into the live book on stale
                # prices) and to have the t+1 bar the entry is priced at.
                if P["pidx"][-1] < dt - np.timedelta64(7, "D") or n_fwd < 1:
                    continue
                full_fwd = False
            elif not full_fwd:
                # Series ends inside the hold window. Normally skip (the stock
                # is unbuyable-through-death in this data); under the delist
                # stress knob include it as a truncated position -- but only if
                # the series truly dies before the sample does (otherwise it's
                # just the end of the data, not a delisting), and only if there
                # is still a t+1 bar to enter on.
                died = P["pidx"][-1] < spx_end - np.timedelta64(7, "D")
                if DELIST_HAIRCUT is None or not died or n_fwd < 1:
                    continue
            mc = asof(P["fidx"], P["mcap"], dlag)
            if mc is None or np.isnan(mc):
                continue
            cand.append((mc, ck, pos, full_fwd))
        if len(cand) < 50:
            continue
        cand.sort(key=lambda x: x[0], reverse=True)
        # With real PIT membership we keep ALL members (membership IS the
        # universe definition); only the proxy needs the top-N-by-mcap cut.
        if not pit_set:
            cand = cand[:universe]
        rows = []
        for mc, ck, pos, full_fwd in cand:
            P = prep[ck]; arr = P["arr"]; w = P["rets"][pos - LB + 1: pos + 1]
            wm = P["mret"][pos - LB + 1: pos + 1]
            sd = w.std(ddof=1)
            ok = ~np.isnan(w) & ~np.isnan(wm); beta = np.nan
            if ok.sum() > 60 and wm[ok].var() > 0:
                beta = float(np.cov(w[ok], wm[ok])[0, 1] / wm[ok].var())
            ret_63 = arr[pos] / arr[pos - 63] - 1
            row = {
                "company_key": ck, "sector": P["sector"], "pit_mcap": mc, "beta": beta,
                "ret_21": arr[pos] / arr[pos - 21] - 1,
                "ret_63": ret_63,
                "ret_126": arr[pos] / arr[pos - 126] - 1,
                # 12-1 momentum (Jegadeesh-Titman): the canonical formulation --
                # 12-month return SKIPPING the most recent month, because that
                # last month carries short-term reversal that works against it.
                # pos >= LB (251) so pos-251 is always in bounds.
                "ret_12_1": arr[pos - 21] / arr[pos - 251] - 1,
                "accel": ret_63 - (arr[pos - 63] / arr[pos - 126] - 1),
                "hi_252": arr[pos] / arr[pos - 251: pos + 1].max(),
                "rs_63": ret_63 - m63,                  # relative strength vs market
                "vol_21": float(w[-21:].std(ddof=1) * np.sqrt(252)) if sd > 0 else np.nan,
                "rev_growth": asof(P["fidx"], P["rgr"], dlag),   # YoY rev growth (90d-lagged)
                # EXECUTION LAG: signal is formed on close t but the trade enters at
                # close t+1 (you can't compute the whole cross-section AND trade the
                # close it was computed from). Validated benign-to-favorable, so the
                # tradeable next-day entry is the honest default. On the LIVE
                # rebalance no window has finished yet; under the delist stress knob
                # a dying name is truncated at its last trade with the haircut
                # applied -- every branch enters at the same t+1 close.
                "fwd_ret": (arr[pos + 1 + hold] / arr[pos + 1] - 1) if full_fwd
                           else (np.nan if is_live
                                 else (arr[-1] * (1.0 + DELIST_HAIRCUT)) / arr[pos + 1] - 1),
            }
            if is_live:                       # mark the open position to the latest close
                row["fwd_todate"] = arr[-1] / arr[pos + 1] - 1
            rows.append(row)
        regime_on = bool(spx.iloc[mpos] > ma200.iloc[mpos]) if not np.isnan(ma200.iloc[mpos]) else True
        if is_live:
            live_rows = pd.DataFrame(rows)
            live_date = pd.Timestamp(d)
            live_regime_on = regime_on
            live_spx_todate = float(spx.iloc[-1] / spx.iloc[mpos] - 1)
            continue
        panels.append(pd.DataFrame(rows)); bdates.append(pd.Timestamp(d))
        # benchmark forward return over the FULL hold-day window (hold-aware!)
        spxf.append(_bench_fwd_hold(spx, d, hold))
        ndxf.append(_bench_fwd_hold(ndx, d, hold))
        ma200_on.append(regime_on)

    sectors = sorted({s for df in panels for s in df["sector"].unique() if str(s).lower() != "unknown"})
    return EdgePanel(panels, pd.DatetimeIndex(bdates), np.array(spxf), np.array(ndxf),
                     np.array(ma200_on, bool), mret_full, sectors, hold,
                     live_panel=live_rows, live_date=live_date,
                     live_regime_on=live_regime_on, live_spx_todate=live_spx_todate)


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


def _scorable(df: pd.DataFrame, require_fwd: bool) -> pd.DataFrame:
    """Rows we're allowed to pick from. The backtest requires a realized forward
    return (a name we couldn't have measured must not enter a basket); the LIVE
    book has no forward return by definition, so it opts out."""
    return (df.dropna(subset=["fwd_ret"]) if require_fwd else df).copy()


def select_topN(df: pd.DataFrame, n: int = N, weights: dict | None = None,
                require_fwd: bool = True) -> list:
    d = _scorable(df, require_fwd)
    d["_c"] = score(d, weights)
    return list(d.sort_values("_c", ascending=False).head(n)["company_key"])


def corr_cap_select(df, asof=None, n=N, weights=None, cap=0.50, lookback=126,
                    require_fwd=True):
    """Greedy basket: walk names best-first, admit a name only if its return
    correlation with every already-admitted name is <= cap.

    `asof` (the rebalance date) is REQUIRED for a point-in-time window -- the
    correlation is judged on the `lookback` trading days ending at `asof`, never
    future data. (Omitting it falls back to the most-recent window, which is only
    valid for a live pick, not a historical backtest.)"""
    d = _scorable(df, require_fwd)
    d["_c"] = score(d, weights)
    order = list(d.sort_values("_c", ascending=False)["company_key"])
    M = D.daily_return_matrix()
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
    M = D.daily_return_matrix(); cal = M.index
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


def sortino(rets, ppy=PPY):
    """Annualized return per unit of DOWNSIDE deviation (MAR=0) — like Sharpe but
    only losses count as risk. Same no-excess-of-cash convention as perf()/the page."""
    r = np.nan_to_num(np.asarray(rets, float))
    if len(r) == 0:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return float(np.sqrt(ppy) * np.mean(r) / dd) if dd > 0 else 0.0


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
    # Match the stock-side execution lag: enter at the first close after d (t+1)
    # and exit hold days later, so the benchmark is measured over the same window.
    fwd = series[series.index > d]
    if len(fwd) < hold + 1:
        return np.nan
    return float(fwd.iloc[hold] / fwd.iloc[0] - 1)


# ---- the robust, tradeable Edge: signal + liquidity + corr-cap + regime + costs
EDGE_SPEC = dict(signal={"accel": 1.0}, hold=42, n=10, mcap_floor=2e9,
                 corr_cap=0.50, corr_lookback=126, regime_expo=0.25, cost_bps=10.0,
                 growth_mix=0.75,          # fraction of the basket drawn from the >=15%-rev-growth pool
                 growth_thresh=0.15,       # YoY revenue-growth bar that defines a "growth" name
                 stagger=True,             # two sleeves offset by half a period (removes rebalance-
                                           # date luck); performance measured on the DAILY curve so
                                           # maxDD reflects true intra-period peak-to-trough.
                 continuous_regime=True)   # evaluate the 200dMA de-risk DAILY across the hold window
                                           # (not frozen at rebalance). Robust drawdown reducer -- helps
                                           # or ties in every crisis, tied Sharpe (2026-07 research);
                                           # catches fast intra-window crashes a per-rebalance regime
                                           # rides through (e.g. COVID-2020: -34% -> -26% at n=10).


def _blend_select(d, asof, n, weights, corr_cap, corr_lookback, growth_mix, growth_thresh,
                  require_fwd=True):
    """Build the basket as K growth-gated picks + (n-K) pure-signal picks, where
    K = round(growth_mix * n). growth_mix=0 -> pure signal; 1 -> all-growth.

    `require_fwd=False` picks from rows with no realized forward return -- only
    valid for the LIVE book (the still-open position), never for the backtest."""
    def pick(frame, k):
        if k <= 0 or len(frame) < 1:
            return []
        return (corr_cap_select(frame, asof=asof, n=k, weights=weights, cap=corr_cap,
                                lookback=corr_lookback, require_fwd=require_fwd) if corr_cap
                else select_topN(frame, n=k, weights=weights, require_fwd=require_fwd))
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


# The heavy layer. Its key space over the PUBLIC selector options is
# hold(4) x n(6) x growth_mix(5) = 120 distinct computations (windows are cheap
# slices of these, so they cost nothing extra). Size the cache ABOVE that space so
# entries are never evicted: total heavy work is then bounded at 120 per worker for
# the life of the process, no matter how many parameter combos get requested.
# Entries are small (a handful of ~250-float arrays), so this costs ~MBs.
@lru_cache(maxsize=160)
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
    # holds (per-rebalance company_key baskets) is returned too so callers like the
    # Edge Tracker can show what was actually held each rebalance, with no recompute.
    return pan.bdates, gross, net, turn, pan.spxf, pan.ndxf, holds


# ---- daily buy-and-hold simulation + staggered sleeves ----------------------
def _select_holds(pan, n, weights, mcap_floor, corr_cap, corr_lookback,
                  growth_mix, growth_thresh):
    """Full-spec per-rebalance selection + membership turnover for one panel."""
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= mcap_floor] if mcap_floor else df
        cks = _blend_select(d, pan.bdates[i], n, weights, corr_cap, corr_lookback,
                            growth_mix, growth_thresh)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn)


@lru_cache(maxsize=64)
def _select_holds_cached(hold, offset_days, n, signal_key, mcap_floor, corr_cap,
                         corr_lookback, growth_mix, growth_thresh):
    """Selection is the SAME for the net and gross backtest passes (cost doesn't
    change which names are picked), and it's the dominant cost (~8s/sleeve via the
    correlation cap). Cache it on the hashable spec so the gross pass — and repeat
    backtests at a different cost/window — reuse it instead of re-selecting."""
    pan = load_edge_panel(hold=hold, offset_days=offset_days)
    holds, turn = _select_holds(pan, n, dict(signal_key), mcap_floor, corr_cap,
                                corr_lookback, growth_mix, growth_thresh)
    return pan, holds, turn


@lru_cache(maxsize=1)
def _ma200_daily_state():
    """Daily boolean on the return-matrix calendar: is the S&P above its 200-day
    MA, judged at the PRIOR close (causal, no look-ahead)? True where the MA isn't
    yet defined (early history) -- matching the per-rebalance regime's convention
    of staying invested when the state is unknown."""
    M = D.daily_return_matrix(); cal = M.index
    spx = D.benchmarks()["SP500"].dropna()
    ma = spx.rolling(200).mean()
    on = pd.Series(np.where(ma.notna(), (spx > ma).to_numpy(), True), index=spx.index)
    on = on.reindex(cal, method="ffill").shift(1).fillna(True)        # prior-close state
    return on.to_numpy(bool)


def _sleeve_daily(pan, holds, turn, M, hold, regime_expo, cost_bps, regime_daily=None):
    """Daily return Series for one sleeve: buy-and-hold each book for `hold` days
    from t+1 (weights drift), regime scales the window, cost on the entry day.
    NaN outside the sleeve's active days so sleeves can be averaged cleanly.

    `regime_daily`: optional daily boolean array (S&P>200dMA, causal) aligned to
    M's calendar. When provided AND regime_expo is set, the de-risk is applied per
    DAY across the hold window (continuous regime) rather than frozen at the
    rebalance date. None -> the original per-rebalance behaviour."""
    cal = M.index
    out = pd.Series(np.nan, index=cal)
    rf_d = config.RISK_FREE_ANNUAL / 252.0
    for i, cks in enumerate(holds):
        loc = int(cal.searchsorted(pan.bdates[i], side="right"))     # t+1 entry (exec lag)
        win = cal[loc: loc + hold]
        names = [c for c in cks if c in M.columns]
        if len(win) == 0 or not names:
            continue
        sub = np.nan_to_num(M.loc[win, names].to_numpy(float))
        w = np.full(len(names), 1.0 / len(names))
        on_rebal = bool(pan.ma200_on[i]) if regime_expo is not None else True
        rets = np.empty(len(win))
        for dd in range(len(win)):
            r = sub[dd]; pr = float(np.dot(w, r))
            if regime_daily is not None and (loc + dd) < len(regime_daily):
                on = bool(regime_daily[loc + dd])        # continuous: this day's state
            else:
                on = on_rebal                            # per-rebalance (or fallback)
            if regime_expo is not None and not on:
                pr = regime_expo * pr + (1 - regime_expo) * rf_d
            rets[dd] = pr
            g = w * (1 + r); ssum = g.sum()
            if ssum > 0:
                w = g / ssum
        rets[0] -= (cost_bps / 1e4) * float(turn[i])                 # entry-day cost
        out.loc[win] = rets
    return out


@lru_cache(maxsize=16)
def _edge_daily(hold, n, mcap_floor, corr_cap, corr_lookback, regime_expo, cost_bps,
                signal_key, growth_mix, growth_thresh, stagger, continuous_regime=False):
    """Daily NET return series for the tradeable Edge (staggered sleeves when
    stagger=True). Returns aligned daily model / S&P / Nasdaq returns + turnover +
    the primary sleeve's holds. maxDD taken on THIS daily curve = true peak-to-trough.

    continuous_regime=True evaluates the 200dMA de-risk per DAY across each hold
    window (a robust drawdown reducer) instead of freezing it at the rebalance."""
    weights = dict(signal_key)
    M = D.daily_return_matrix()
    rd = _ma200_daily_state() if (continuous_regime and regime_expo is not None) else None
    offsets = (0, hold // 2) if stagger else (0,)
    series, turns, prim_holds = [], [], None
    for off in offsets:
        pan, holds, turn = _select_holds_cached(hold, off, n, signal_key, mcap_floor,
                                                corr_cap, corr_lookback, growth_mix, growth_thresh)
        series.append(_sleeve_daily(pan, holds, turn, M, hold, regime_expo, cost_bps, rd))
        turns.append(float(np.mean(turn)))
        if off == 0:
            prim_holds = (pan.bdates, holds)
    both = pd.concat(series, axis=1)
    model = both.mean(axis=1, skipna=True).dropna()      # 50/50 where both active
    idx = model.index
    bm = D.benchmarks()
    spx = bm["SP500"].reindex(idx, method="ffill").pct_change().fillna(0.0)
    ndx = bm["NASDAQ"].reindex(idx, method="ffill").pct_change().fillna(0.0)
    return (idx, model.to_numpy(), spx.to_numpy(), ndx.to_numpy(),
            float(np.mean(turns)), prim_holds)


def _perf_daily(r):
    """CAGR / maxDD / Sharpe from a DAILY net-return array (annualized at 252)."""
    r = np.nan_to_num(np.asarray(r, float))
    if len(r) < 2:
        return 0.0, 0.0, 0.0
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (252.0 / len(r)) - 1 if eq[-1] > 0 else -1.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sh = float(np.sqrt(252) * np.nanmean(r) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
    return float(cagr), dd, sh


def _sortino_daily(r):
    r = np.nan_to_num(np.asarray(r, float))
    dn = np.minimum(r, 0.0); d = float(np.sqrt(np.mean(dn ** 2)))
    return float(np.sqrt(252) * np.mean(r) / d) if d > 0 else 0.0


def reset_caches() -> None:
    """Drop every memoized result that depends on the panel.

    REQUIRED after mutating the module-level research knobs USE_PIT_UNIVERSE or
    DELIST_HAIRCUT: neither is part of any cache key, so a surviving entry
    silently returns the PREVIOUS variant's numbers -- a wrong research result
    with no error. Clearing load_edge_panel alone is not enough; the selection
    and daily-series caches sit above it and would still be warm.
    (_ma200_daily_state is not cleared: it depends only on the S&P calendar.)"""
    for fn in (load_edge_panel, _select_holds_cached, _edge_full, _edge_daily):
        fn.cache_clear()


def run_edge_backtest(window: str = "MAX", spec: dict | None = None) -> dict:
    """Backtest the tradeable Edge over a trailing window. Returns curves +
    performance in the same shape the website's chart code expects (NET of costs)."""
    s = {**EDGE_SPEC, **(spec or {})}
    hold = s["hold"]
    stagger = bool(s.get("stagger", True))
    sig = tuple(sorted(s["signal"].items()))
    gm = float(s.get("growth_mix", 0.0) or 0.0); gt = float(s.get("growth_thresh", 0.15))
    cr = bool(s.get("continuous_regime", False))
    idx, model, spx, ndx, avg_to, prim = _edge_daily(
        hold, s["n"], s["mcap_floor"], s["corr_cap"], s["corr_lookback"],
        s["regime_expo"], s["cost_bps"], sig, gm, gt, stagger, cr)
    # gross (cost-free) daily model, for the gross->net turnover card
    gidx, gmodel, *_ = _edge_daily(hold, s["n"], s["mcap_floor"], s["corr_cap"],
                                   s["corr_lookback"], s["regime_expo"], 0.0, sig, gm, gt, stagger, cr)
    T = len(model)
    k = window_k(window, T)
    if k < 20:
        return {"ok": False, "reason": "window too short"}
    sl = slice(T - k, T)
    mr, sr, nr = model[sl], spx[sl], ndx[sl]
    dts = idx[T - k:T]

    def curve(r):                                    # growth-of-$1, subsampled for payload
        eq = np.cumprod(1 + np.nan_to_num(r))
        stride = max(1, len(eq) // 300)
        keep = list(range(0, len(eq), stride))
        if keep[-1] != len(eq) - 1:
            keep.append(len(eq) - 1)
        return keep, [round(float(eq[i]), 4) for i in keep]

    def stats(r):
        c, dd, sh = _perf_daily(r)
        tot = float(np.cumprod(1 + np.nan_to_num(r))[-1] - 1)
        return {"cagr": c, "sharpe": sh, "sortino": _sortino_daily(r),
                "max_drawdown": dd, "total_return": tot}

    windows = []
    for wn in SUMMARY_WINDOWS:
        kk = window_k(wn, T)
        cg, dd, sh = _perf_daily(model[T - kk:T]); cs, _, _ = _perf_daily(spx[T - kk:T])
        windows.append({"w": wn, "cagr": cg, "sharpe": sh, "dd": dd,
                        "sp_cagr": cs, "excess": cg - cs})
    gc = _perf_daily(gmodel[len(gmodel) - min(k, len(gmodel)):])[0]

    # best / worst single hold-length window (rolling), from the daily net series
    mser = pd.Series(mr, index=dts)
    roll = (1 + mser).rolling(hold).apply(np.prod, raw=True) - 1
    roll = roll.dropna()
    bi = roll.idxmax(); wi = roll.idxmin()
    period_detail = {
        "hold_days": hold,
        "best": {"ret": float(roll.max()), "date": str(bi.date())},
        "worst": {"ret": float(roll.min()), "date": str(wi.date())},
        "pretax_total": stats(mr)["total_return"],   # net of costs, before taxes
        "pretax_cagr": stats(mr)["cagr"],
    }
    keep, mcurve = curve(mr)
    dstr = [str(dts[i].date()) for i in keep]
    ann_to = round(avg_to * (252.0 / hold), 1)
    return {
        "ok": True, "n_rebalances": int(k / hold),
        "period": [str(dts[0].date()), str(dts[-1].date())],
        "period_detail": period_detail,
        "spec": {"signal": "acceleration (3m−prior3m)", "hold_days": hold,
                 "n": s["n"], "mcap_floor_bn": s["mcap_floor"] / 1e9,
                 "corr_cap": s["corr_cap"], "regime_expo": s["regime_expo"],
                 "cost_bps": s["cost_bps"], "stagger": stagger,
                 "growth_mix": gm, "growth_thresh": gt, "continuous_regime": cr},
        "performance": {"model": stats(mr), "sp500": stats(sr), "nasdaq": stats(nr)},
        "curves": {"dates": dstr, "model": mcurve,
                   "sp500": curve(sr)[1], "nasdaq": curve(nr)[1]},
        "windows": windows,
        "turnover": {"per_rebalance": round(avg_to, 3), "annualized": ann_to,
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
