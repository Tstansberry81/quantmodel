"""Shared engine for the tech-bias / alpha-attribution research scripts.

Builds the monthly rebalance panel once (the slow part: ~250 periods over the
~500-name point-in-time universe, 2005-2026), then exposes helpers to:
  * build the strategy's model returns for any param set (with optional
    vol-scaling),
  * construct in-universe factor returns (market, size, value, momentum) at the
    strategy's OWN rebalance frequency (avoids calendar-alignment error),
  * run an OLS regression with Newey-West (HAC) standard errors.

In-universe factors vs. published Fama-French: building size/value/momentum as
long-short tercile portfolios from the SAME universe the strategy picks from,
at the same 21-day frequency, is the cleaner apples-to-apples test of "does my
selection add value beyond these exposures within my own pond." A cross-check
against published Ken-French factors (calendar-aligned) is a separate validation.

No statsmodels in the venv -> OLS / Newey-West are done by hand in numpy.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from functools import lru_cache
import numpy as np, pandas as pd
import config
from qmodel import engine, scoring, backtest as bt
from qmodel.equations import FACTORS

N = config.N_STOCKS
HOLD = 21                      # rebalance + holding spacing in trading days (~monthly)
LB = 251                       # trailing window for price factors
SKIP = 21                      # skip most-recent month (12-1 momentum)
PPY = 252.0 / HOLD             # rebalances per year (~12)
RF_PER = config.RISK_FREE_ANNUAL / PPY
FILING_LAG = np.timedelta64(bt.FILING_LAG_DAYS, "D")

# value ratios pulled for the HML (value) factor
VAL_IDS = {"earnings_yield": "ratio_earnings_yield", "fcf_yield": "ratio_fcf_yield"}


def _asof(fidx, varr, dlag):
    if fidx is None or varr is None:
        return np.nan
    p = int(np.searchsorted(fidx, dlag, side="right")) - 1
    return varr[p] if p >= 0 else np.nan


def _bench_fwd(series, d):
    past = series[series.index <= d]; fwd = series[series.index > d]
    if past.empty or len(fwd) < 1:
        return np.nan
    return float(fwd.iloc[min(HOLD, len(fwd)) - 1] / past.iloc[-1] - 1)


class Panel:
    """Holds the prebuilt rebalance panel + market/benchmark series."""
    def __init__(self, panels, bdates, spxf, ndxf, sector_ret, sectors):
        self.panels = panels            # list[DataFrame] per rebalance
        self.bdates = bdates            # DatetimeIndex
        self.spxf = spxf                # np.array  S&P fwd return per period
        self.ndxf = ndxf                # np.array  Nasdaq fwd return per period
        self.sector_ret = sector_ret    # list[dict] cap-wtd sector fwd returns
        self.sectors = sectors          # sorted sector names (excl. "Unknown")
        self.T = len(panels)


@lru_cache(maxsize=1)
def load_panel() -> Panel:
    """Build the rebalance panel once (cached per process)."""
    data = engine._load_bt_data(); bm = engine._benchmarks_cached()
    mkt = bm["SP500"].dropna(); mret_full = mkt.pct_change()
    ndx = bm["NASDAQ"].dropna()
    fund_ids = {f: FACTORS[f]["ratio_id"] for f in FACTORS if FACTORS[f]["ratio_id"]}

    prep = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB + HOLD + 2:
            continue
        arr = np.asarray(pr.values, float); pidx = np.asarray(pr.index.values, "datetime64[ns]")
        rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:] / arr[:-1] - 1
        mret = np.asarray(mret_full.reindex(pr.index, method="ffill").values, float)
        fh = blob.get("fund_hist"); fidx = None; fcols = {}; vcols = {}; mcap = None
        if fh is not None and not fh.empty:
            fidx = np.asarray(fh.index.values, "datetime64[ns]")
            for fn, rid in fund_ids.items():
                fcols[fn] = np.asarray(fh[rid].values, float) if rid in fh.columns else None
            for vn, rid in VAL_IDS.items():
                vcols[vn] = np.asarray(fh[rid].values, float) if rid in fh.columns else None
            if "calculated_market_cap" in fh.columns:
                mcap = np.asarray(fh["calculated_market_cap"].values, float)
        prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret, "fidx": fidx,
                    "fcols": fcols, "vcols": vcols, "mcap": mcap,
                    "sector": blob.get("meta", {}).get("sector", "Unknown")}

    spx = mkt
    monthly = spx.index[spx.index >= spx.index.min() + pd.Timedelta(days=400)]
    rebal = list(monthly[:-HOLD][::HOLD])
    panels, bdates, spxf, ndxf, sector_ret = [], [], [], [], []

    for d in rebal:
        dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt - FILING_LAG
        cand = []
        for ck, P in prep.items():
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB or (len(P["arr"]) - 1 - pos) < HOLD:
                continue
            mc = _asof(P["fidx"], P["mcap"], dlag)
            if mc is None or np.isnan(mc):
                continue
            cand.append((mc, ck, pos))
        if len(cand) < 50:
            continue
        cand.sort(key=lambda x: x[0], reverse=True); cand = cand[:config.BT_UNIVERSE_SIZE]
        rows = []
        for mc, ck, pos in cand:
            P = prep[ck]; arr = P["arr"]
            w = P["rets"][pos - LB + 1:pos + 1]; wm = P["mret"][pos - LB + 1:pos + 1]
            sd = w.std(ddof=1); neg = w[w < 0]; dd = neg.std(ddof=1) if neg.size > 1 else np.nan
            p6, p12 = arr[pos - 125], arr[pos - 251]; hi = arr[pos - 251:pos + 1].max()
            ok = ~np.isnan(w) & ~np.isnan(wm); rm = np.nan; beta = np.nan
            if ok.sum() > 60:
                rr, mm = w[ok], wm[ok]
                if mm.var() > 0:
                    beta = float(np.cov(rr, mm)[0, 1] / mm.var())
                    resid = rr - (rr.mean() - beta * mm.mean()) - beta * mm
                    rv = resid.std(ddof=1)
                    if rv > 0 and resid.size > SKIP:
                        rm = resid[:-SKIP].sum() / rv
            row = {"company_key": ck, "sector": P["sector"], "pit_mcap": mc, "beta": beta,
                   "momentum": arr[pos - 20] / arr[pos - 251] - 1, "resid_mom": rm,
                   "mom_accel": ((arr[pos] / p6 - 1) - (p6 / p12 - 1)) if p6 > 0 and p12 > 0 else np.nan,
                   "mom_52w_high": arr[pos] / hi if hi > 0 else np.nan,
                   "sharpe": (np.sqrt(252) * (w.mean() - config.RISK_FREE_ANNUAL / 252) / sd) if sd > 0 else np.nan,
                   "sortino": (np.sqrt(252) * (w.mean() - config.RISK_FREE_ANNUAL / 252) / dd) if dd and dd > 0 else np.nan,
                   "volatility": np.sqrt(252) * sd, "reversal_1m": arr[pos] / arr[pos - 20] - 1,
                   "earnings_yield": _asof(P["fidx"], P["vcols"].get("earnings_yield"), dlag),
                   "fcf_yield": _asof(P["fidx"], P["vcols"].get("fcf_yield"), dlag),
                   "fwd_ret": arr[pos + HOLD] / arr[pos] - 1}
            for fn in fund_ids:
                row[fn] = _asof(P["fidx"], P["fcols"].get(fn), dlag)
            rows.append(row)
        df = pd.DataFrame(rows)
        sr = {}
        for sec, g in df.dropna(subset=["fwd_ret"]).groupby("sector"):
            wcap = g["pit_mcap"].to_numpy(float)
            if wcap.sum() > 0:
                sr[sec] = float(np.average(g["fwd_ret"].to_numpy(float), weights=wcap))
        panels.append(df); sector_ret.append(sr); bdates.append(pd.Timestamp(d))
        spxf.append(_bench_fwd(spx, d)); ndxf.append(_bench_fwd(ndx, d))

    sectors = sorted({s for sr in sector_ret for s in sr if s.lower() != "unknown"})
    return Panel(panels, pd.DatetimeIndex(bdates), np.array(spxf), np.array(ndxf),
                 sector_ret, sectors)


# ---- strategy model returns -------------------------------------------------
def model_returns(pan: Panel, params: dict, vol_scale: bool = False,
                  target_vol: float | None = None, lev_cap: float = 1.0,
                  vol_win: int = 6, cost_bps: float = 0.0):
    """Top-N equal-weight model returns for a param set.

    vol_scale: Barroso-Santa-Clara -- scale each period's exposure by
        target_vol / trailing realized vol of the strategy's own returns; the
        un-invested fraction earns the risk-free rate.
    cost_bps: one-way transaction cost in basis points, charged on turnover.
    Returns dict with arrays: ret, tech_w, beta, turnover.
    """
    mret, tech_w, bbeta, turn = [], [], [], []
    hard = params["settings"].get("hard_filters", []) or []
    min_g = FACTORS["rev_growth"]["threshold"] if "rev_growth" in hard else None
    prev = set()
    for df in pan.panels:
        cs = scoring.compute_scores(df, params)
        valid = cs.dropna(subset=["composite", "fwd_ret"]).copy()
        pool = valid
        if min_g is not None:
            gg = pd.to_numeric(valid["rev_growth"], errors="coerce")
            sub = valid[gg >= min_g]
            pool = sub if len(sub) >= N else valid
        top = pool.sort_values("composite", ascending=False).head(N)
        cur = set(top["company_key"])
        to = 1.0 - len(cur & prev) / len(cur) if cur and prev else (1.0 if cur else 0.0)
        prev = cur
        r = float(top["fwd_ret"].mean()) - (cost_bps / 1e4) * to
        mret.append(r); turn.append(to)
        tw = top["sector"].str.lower().str.contains(
            "information technology|communication", regex=True, na=False).mean()
        tech_w.append(float(tw)); bbeta.append(float(top["beta"].mean()))
    mret = np.array(mret)
    if vol_scale:
        tv = target_vol if target_vol is not None else np.nanstd(mret) * np.sqrt(PPY)
        scaled = mret.copy()
        for t in range(len(mret)):
            if t < vol_win:
                continue
            rv = np.nanstd(mret[t - vol_win:t]) * np.sqrt(PPY)
            lev = min((tv / rv) if rv > 0 else 1.0, lev_cap)
            scaled[t] = lev * mret[t] + (1 - lev) * RF_PER
        mret = scaled
    return {"ret": mret, "tech_w": np.array(tech_w), "beta": np.array(bbeta),
            "turnover": np.array(turn)}


# ---- holdings + daily strategy returns (for vol-scaling / external checks) --
def holdings(pan: Panel, params: dict):
    """The list of selected company_keys at each rebalance (same selection as
    model_returns: compute scores -> rev_growth hard filter -> top-N composite)."""
    hard = params["settings"].get("hard_filters", []) or []
    min_g = FACTORS["rev_growth"]["threshold"] if "rev_growth" in hard else None
    out = []
    for df in pan.panels:
        cs = scoring.compute_scores(df, params)
        valid = cs.dropna(subset=["composite", "fwd_ret"]).copy()
        pool = valid
        if min_g is not None:
            gg = pd.to_numeric(valid["rev_growth"], errors="coerce")
            sub = valid[gg >= min_g]
            pool = sub if len(sub) >= N else valid
        top = pool.sort_values("composite", ascending=False).head(N)
        out.append(list(top["company_key"]))
    return out


@lru_cache(maxsize=1)
def daily_return_matrix():
    """DataFrame of daily returns for every name, reindexed to the S&P trading
    calendar (forward-filled prices). Columns = company_key."""
    data = engine._load_bt_data(); bm = engine._benchmarks_cached()
    cal = bm["SP500"].dropna().index
    cols = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB:
            continue
        cols[ck] = pr.reindex(cal, method="ffill").pct_change().values
    return pd.DataFrame(cols, index=cal)


def daily_strategy_returns(pan: Panel, params: dict) -> pd.Series:
    """Daily equal-weight return series of the strategy, rebalanced every HOLD
    trading days into the selected top-N (daily-rebalanced within each period).
    Use this for daily vol-scaling (Barroso-Santa-Clara) or calendar-month
    aggregation against published factors."""
    M = daily_return_matrix(); cal = M.index
    hold = holdings(pan, params)
    pieces = []
    for i, d in enumerate(pan.bdates):
        cks = [c for c in hold[i] if c in M.columns]
        loc = int(cal.searchsorted(d, side="right"))
        win = cal[loc: loc + HOLD]
        if len(win) == 0 or not cks:
            continue
        pieces.append(M.loc[win, cks].mean(axis=1))
    if not pieces:
        return pd.Series(dtype=float)
    s = pd.concat(pieces)
    return s[~s.index.duplicated(keep="first")].fillna(0.0)


# ---- in-universe factor returns (size / value / momentum) -------------------
def _ls_factor(df: pd.DataFrame, col: str, higher_is_long: bool = True) -> float:
    """Long-short return: top tercile minus bottom tercile (equal-weighted),
    ranked by `col`, using each name's forward return."""
    g = df.dropna(subset=[col, "fwd_ret"])
    if len(g) < 15:
        return np.nan
    v = pd.to_numeric(g[col], errors="coerce")
    f = g["fwd_ret"].to_numpy(float)
    q1, q2 = v.quantile(1/3), v.quantile(2/3)
    hi = f[(v >= q2).to_numpy()]; lo = f[(v <= q1).to_numpy()]
    if len(hi) == 0 or len(lo) == 0:
        return np.nan
    long_leg, short_leg = (hi, lo) if higher_is_long else (lo, hi)
    return float(np.nanmean(long_leg) - np.nanmean(short_leg))


def carhart_factors(pan: Panel) -> dict:
    """Build market/size/value/momentum factor return series at the strategy's
    own frequency, from the same point-in-time universe.

      MKT_RF = S&P fwd return - risk-free
      SMB    = small-cap tercile  - big-cap tercile      (size)
      HML    = high-earnings-yield tercile - low         (value)
      UMD    = high-12-1-momentum tercile - low          (momentum)
    """
    smb, hml, umd = [], [], []
    for df in pan.panels:
        smb.append(_ls_factor(df, "pit_mcap", higher_is_long=False))   # small minus big
        val = "earnings_yield" if df["earnings_yield"].notna().sum() >= 15 else "fcf_yield"
        hml.append(_ls_factor(df, val, higher_is_long=True))           # cheap minus expensive
        umd.append(_ls_factor(df, "momentum", higher_is_long=True))    # winners minus losers
    return {"MKT_RF": pan.spxf - RF_PER,
            "SMB": np.nan_to_num(np.array(smb)),
            "HML": np.nan_to_num(np.array(hml)),
            "UMD": np.nan_to_num(np.array(umd))}


def sector_tilts(pan: Panel) -> dict:
    """Sector-tilt factors = each sector's cap-wtd return minus the market.
    Drops one sector (cap-wtd sectors sum to ~market -> avoid the dummy trap)."""
    out = {}
    for s in pan.sectors[:-1]:
        v = np.array([sr.get(s, pan.spxf[i]) for i, sr in enumerate(pan.sector_ret)])
        out[f"tilt_{s[:10]}"] = v - pan.spxf
    return out


# ---- OLS + Newey-West -------------------------------------------------------
def ols_nw(y, X, L: int = 6):
    """OLS with Newey-West HAC standard errors. X must include the constant."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        wgt = 1.0 - l / (L + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se


def attribution(label, model_ret, regressors: dict, mask=None, verbose=True):
    """Regress model EXCESS return on the given regressors (dict name->array).
    Returns (annualized_alpha, alpha_t, dict of factor betas/t). Prints if verbose."""
    y = model_ret - RF_PER
    names = ["alpha"] + list(regressors)
    cols = [np.ones(len(y))] + [regressors[k] for k in regressors]
    X = np.column_stack(cols)
    if mask is not None:
        y, X = y[mask], X[mask]
    beta, se, t = ols_nw(y, X)
    a_ann = (1 + beta[0]) ** PPY - 1
    if verbose:
        tag = "** survives" if abs(t[0]) > 2 else "NOT significant" if abs(t[0]) < 1.65 else "~ marginal"
        print(f"  [{label}]   alpha {a_ann*100:+.2f}%/yr   t={t[0]:+.2f}   {tag}")
        for i, nm in enumerate(names[1:], 1):
            print(f"        {nm:<16} loading={beta[i]:+.2f}  t={t[i]:+.2f}")
    return a_ann, float(t[0]), {nm: (float(beta[i]), float(t[i])) for i, nm in enumerate(names)}


def perf(mret, ppy=PPY):
    """(CAGR, maxDD, Sharpe) for a per-period return array."""
    eq = np.cumprod(1 + np.nan_to_num(mret))
    cagr = eq[-1] ** (ppy / len(mret)) - 1 if eq[-1] > 0 else -1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    shp = float(np.sqrt(ppy) * np.nanmean(mret) / np.nanstd(mret))
    return float(cagr), dd, shp
