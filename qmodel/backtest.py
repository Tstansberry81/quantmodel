"""Honest cross-sectional backtest.

At each monthly rebalance we reconstruct the composite using only data known at
that time (fundamentals lagged to their filing date; price factors as-of the
date), then measure:
  * IC      - Spearman rank corr of score vs forward 1M return, each period
  * IC IR   - mean(IC)/std(IC), and a t-stat
  * deciles - mean forward return by score decile (monotonicity check)
  * curve   - equity curve of the top-N model portfolio vs S&P 500 & Nasdaq

Caveat surfaced in the UI: the HMM regime used for the gold sleeve is fit once
on the full sample, so the gold overlay carries mild lookahead; the equity
signal itself (the part that matters for IC) is strictly point-in-time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from qmodel import scoring, factors as factmod, optimize, hmm_regime
from qmodel.equations import FACTORS

FILING_LAG_DAYS = 90       # assume annual fundamentals known ~90d after period end
FWD_DAYS = 21               # ~1 trading month forward window


def _fund_asof(fund_hist: pd.DataFrame, date: pd.Timestamp, ratio_id: str):
    """Most recent fundamental value known at `date` (report date + filing lag)."""
    if fund_hist is None or fund_hist.empty or ratio_id not in fund_hist.columns:
        return np.nan
    avail = fund_hist[fund_hist.index <= (date - pd.Timedelta(days=FILING_LAG_DAYS))]
    if avail.empty:
        return np.nan
    return avail[ratio_id].iloc[-1]


def run_backtest(data: dict, benchmarks: dict, params: dict,
                 start: str | None = None, hold_days: int = 21) -> dict:
    """data: {company_key: {"prices": Series, "fund_hist": DataFrame, "meta": {...}}}

    hold_days sets both the holding period and the rebalance spacing in trading
    days (5=weekly, 10=biweekly, 21=monthly, 63=quarterly, ...), so periods are
    non-overlapping.
    """
    hold_days = max(2, int(hold_days))
    ppy = 252.0 / hold_days                        # rebalances per year (for annualizing)

    # rebalance calendar = every hold_days-th S&P 500 trading day (with room for the forward window)
    spx = benchmarks["SP500"].dropna()
    if spx.empty:
        return {"ok": False, "reason": "no benchmark data"}
    tdays = spx.index
    lo = pd.Timestamp(start) if start else (tdays.min() + pd.Timedelta(days=400))
    tdays = tdays[tdays >= lo]
    usable = tdays[:-hold_days] if len(tdays) > hold_days else tdays[:0]
    rebal_dates = list(usable[::hold_days])
    if len(rebal_dates) < 2:
        return {"ok": False, "reason": (
            f"Lookback window is too short for this holding period "
            f"(only {len(rebal_dates)} rebalance fits). Pick a longer window or a shorter holding period.")}

    fund_factor_ids = {f: FACTORS[f]["ratio_id"] for f in FACTORS if FACTORS[f]["ratio_id"]}
    n_inactive = sum(1 for b in data.values()
                     if b.get("meta", {}).get("trading_status", "Active") != "Active")

    # ---- precompute per-name numpy arrays once (the loop below is hot) ----
    LB = 251                                  # trailing return-window length
    rf_daily = config.RISK_FREE_ANNUAL / config.TRADING_DAYS
    sqrtT = float(np.sqrt(config.TRADING_DAYS))
    lag = np.timedelta64(FILING_LAG_DAYS, "D")
    SKIP = 21                                    # skip most-recent month (12-1)
    mkt_ret = benchmarks["SP500"].dropna().pct_change()
    prep = {}
    for ck, blob in data.items():
        prices = blob.get("prices")
        if prices is None or len(prices) < LB + hold_days + 2:
            continue
        arr = np.asarray(prices.values, dtype=float)
        pidx = np.asarray(prices.index.values, dtype="datetime64[ns]")
        rets = np.empty(len(arr)); rets[0] = np.nan
        rets[1:] = arr[1:] / arr[:-1] - 1.0
        mret = np.asarray(mkt_ret.reindex(prices.index, method="ffill").values, dtype=float)
        fh = blob.get("fund_hist")
        fidx = None; fund_cols = {}; mcap = None
        if fh is not None and not fh.empty:
            fidx = np.asarray(fh.index.values, dtype="datetime64[ns]")
            for fn, rid in fund_factor_ids.items():
                fund_cols[fn] = np.asarray(fh[rid].values, dtype=float) if rid in fh.columns else None
            if "calculated_market_cap" in fh.columns:
                mcap = np.asarray(fh["calculated_market_cap"].values, dtype=float)
        prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret, "fidx": fidx,
                    "fund": fund_cols, "mcap": mcap,
                    "sector": blob.get("meta", {}).get("sector", "Unknown")}

    def _asof(fidx, varr, dt_lagged):
        if fidx is None or varr is None:
            return np.nan
        p = int(np.searchsorted(fidx, dt_lagged, side="right")) - 1
        return varr[p] if p >= 0 else np.nan

    ic_records = []
    decile_rets = {i: [] for i in range(10)}
    model_rets, spx_rets, ndx_rets = [], [], []
    used_dates = []
    prev_top, turnovers = None, []   # portfolio turnover per rebalance
    daily_chunks = []                # Stage 6: per-period daily book returns (for vol-targeting)

    spx_s, ndx_s = benchmarks["SP500"], benchmarks["NASDAQ"]
    gold_s = benchmarks.get("GOLD")

    # Stage 3/4 config: weight scheme + HMM regime overlay
    st = params["settings"]
    scheme = st.get("weight_scheme", "equal")
    cov_w = int(st.get("cov_window", 126))
    wmax = float(st.get("max_weight", 0.30))
    regime_overlay = bool(st.get("regime_overlay", False))
    band = int(st.get("turnover_band", 0) or 0)        # Stage 5: hold-band (0=off)
    vt = bool(st.get("vol_target", False))             # Stage 6: daily vol-targeting
    exp_map = {"bull": float(st.get("exposure_bull", 1.0)),
               "neutral": float(st.get("exposure_neutral", 0.85)),
               "bear": float(st.get("exposure_bear", 0.50))}
    regime_series = None
    if regime_overlay:
        rg = hmm_regime.fit_regime(spx_s, n_states=int(st.get("hmm_states", 3)))
        regime_series = rg.get("regime_series") if rg.get("ok") else None

    for d in rebal_dates:
        dt = np.datetime64(pd.Timestamp(d), "ns")
        dlag = dt - lag
        # cheap pass: point-in-time market cap + valid price position
        cands = []
        for ck, P in prep.items():
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB or (len(P["arr"]) - 1 - pos) < hold_days:
                continue
            mc = _asof(P["fidx"], P["mcap"], dlag)
            if mc is None or np.isnan(mc):
                continue
            cands.append((mc, ck, pos))
        if len(cands) < 20:
            continue
        # Point-in-time universe: hold what was actually largest as of this date.
        if len(cands) > config.BT_UNIVERSE_SIZE:
            cands.sort(key=lambda x: x[0], reverse=True)
            cands = cands[:config.BT_UNIVERSE_SIZE]
        pos_map = {ck: pos for _, ck, pos in cands}

        # expensive pass: numpy-only factor math, top-N names only
        rows = []
        for mc, ck, pos in cands:
            P = prep[ck]; arr = P["arr"]; w = P["rets"][pos - LB + 1: pos + 1]
            sd = w.std(ddof=1)
            mean_ex = w.mean() - rf_daily
            neg = w[w < 0]
            dd = neg.std(ddof=1) if neg.size > 1 else np.nan
            p6, p12 = arr[pos - 125], arr[pos - 251]
            hi52 = arr[pos - 251:pos + 1].max()
            # risk-adjusted residual momentum (Blitz-Huij-Martens): 12-1 cumulative
            # residual return from a market regression, scaled by residual vol.
            wm = P["mret"][pos - LB + 1: pos + 1]
            ok = ~np.isnan(w) & ~np.isnan(wm)
            resid_mom = np.nan
            if ok.sum() > 60:
                rr, mm = w[ok], wm[ok]
                mv = mm.var()
                if mv > 0:
                    beta = np.cov(rr, mm)[0, 1] / mv
                    resid = rr - (rr.mean() - beta * mm.mean()) - beta * mm
                    rvol = resid.std(ddof=1)
                    if rvol > 0 and resid.size > SKIP:
                        resid_mom = resid[:-SKIP].sum() / rvol
            row = {
                "company_key": ck, "sector": P["sector"], "pit_mcap": mc,
                "momentum": arr[pos - 20] / arr[pos - 251] - 1.0,
                "reversal_1m": arr[pos] / arr[pos - 20] - 1.0,
                "resid_mom": resid_mom,
                "mom_accel": ((arr[pos] / p6 - 1.0) - (p6 / p12 - 1.0)) if p6 > 0 and p12 > 0 else np.nan,
                "mom_52w_high": (arr[pos] / hi52) if hi52 > 0 else np.nan,
                "sharpe": (sqrtT * mean_ex / sd) if sd > 0 else np.nan,
                "sortino": (sqrtT * mean_ex / dd) if dd and dd > 0 else np.nan,
                "volatility": sqrtT * sd,
                "fwd_ret": arr[pos + hold_days] / arr[pos] - 1.0,
            }
            for fn in fund_factor_ids:
                row[fn] = _asof(P["fidx"], P["fund"].get(fn), dlag)
            rows.append(row)

        cs = pd.DataFrame(rows)
        cs = scoring.compute_scores(cs, params)

        valid = cs.dropna(subset=["composite", "fwd_ret"])
        if len(valid) < 20 or valid["composite"].nunique() < 2:
            continue

        ic, _ = spearmanr(valid["composite"], valid["fwd_ret"])
        if not np.isnan(ic):
            ic_records.append({"date": d, "ic": ic, "n": len(valid)})

        # deciles
        try:
            valid = valid.assign(decile=pd.qcut(valid["composite"].rank(method="first"),
                                                10, labels=False))
            for dec, g in valid.groupby("decile"):
                decile_rets[int(dec)].append(g["fwd_ret"].mean())
        except ValueError:
            pass

        # ---- Stage 2-6: hard filters -> rank -> (band) -> weights -> overlays ----
        pool = scoring.apply_hard_filters(valid, params)
        ranked = pool.sort_values("composite", ascending=False)
        if band and prev_top:
            # Stage 5 hold-band: keep names still inside the top-`band`, then fill
            # the remaining slots from the highest-ranked names not already held.
            order = list(ranked["company_key"])
            rank_of = {ck: i for i, ck in enumerate(order)}
            sel = [ck for ck in order if ck in prev_top and rank_of[ck] < band]
            for ck in order:
                if len(sel) >= config.N_STOCKS:
                    break
                if ck not in sel:
                    sel.append(ck)
            top = ranked.set_index("company_key").loc[sel[:config.N_STOCKS]].reset_index()
        else:
            top = ranked.head(config.N_STOCKS)
        cks = list(top["company_key"])
        fwd = top["fwd_ret"].to_numpy(float)
        if scheme == "equal":
            w = optimize.equal_weights(len(cks))
        else:
            # covariance from trailing returns of the selected names
            R = np.vstack([prep[ck]["rets"][pos_map[ck] - cov_w + 1: pos_map[ck] + 1] for ck in cks])
            cov = np.cov(R) if len(cks) > 1 else np.array([[1.0]])
            w = optimize.weights_for(scheme, top["composite"].to_numpy(float), cov, wmax)
        equity_ret = float(w @ fwd)

        # Stage 4: HMM regime scales equity exposure; remainder earns gold.
        # (Skipped when Stage-6 vol-targeting is on -- vol-targeting handles exposure.)
        if regime_series is not None and not vt:
            reg = regime_series.asof(d)
            exposure = exp_map.get(reg if isinstance(reg, str) else "neutral", 0.85)
            gfwd = _bench_fwd(gold_s, d, hold_days) if gold_s is not None else np.nan
            if np.isnan(gfwd):
                gfwd = 0.0
            model_rets.append(exposure * equity_ret + (1.0 - exposure) * gfwd)
        else:
            model_rets.append(equity_ret)

        # Stage 6: record the book's *daily* returns over the holding window so we
        # can vol-target (Barroso-Santa-Clara) after the loop.
        if vt:
            segs = []
            for ck in cks:
                p0 = pos_map[ck]
                a = prep[ck]["arr"][p0: p0 + hold_days + 1]
                seg = np.zeros(hold_days)
                if len(a) >= 2:
                    rr = a[1:] / a[:-1] - 1.0
                    seg[:len(rr)] = np.nan_to_num(rr)
                segs.append(seg)
            daily_chunks.append(np.nansum(np.vstack(segs) * w[:, None], axis=0))

        # one-way turnover = fraction of the book that changed vs last rebalance
        cur_top = set(cks)
        if prev_top is not None and cur_top:
            turnovers.append(1.0 - len(cur_top & prev_top) / len(cur_top))
        prev_top = cur_top

        # benchmark forward returns over same window
        spx_rets.append(_bench_fwd(spx_s, d, hold_days))
        ndx_rets.append(_bench_fwd(ndx_s, d, hold_days))
        used_dates.append(d)

    # Stage 6: replace per-period returns with the daily vol-targeted series.
    if vt and daily_chunks:
        model_rets = _vol_target(daily_chunks, st)

    res = _summarize(ic_records, decile_rets, model_rets, spx_rets, ndx_rets, used_dates, ppy)
    if isinstance(res, dict):
        avg_n = float(pd.Series([r["n"] for r in ic_records]).mean()) if ic_records else 0
        res["pool"] = {"n_total": len(data), "n_inactive": n_inactive,
                       "bt_universe_size": config.BT_UNIVERSE_SIZE,
                       "avg_cross_section": round(avg_n, 0)}
        res["hold_days"] = hold_days
        res["n_holdings"] = config.N_STOCKS
        avg_to = float(np.mean(turnovers)) if turnovers else 0.0
        ann_to = avg_to * ppy                       # one-way turnover per year
        # rough net-of-cost CAGR drag: 2*annual one-way turnover * 10bps round-trip
        cost_drag = ann_to * 2 * 0.0010
        mc = res["performance"]["model"]["cagr"]
        res["turnover"] = {
            "per_rebalance": round(avg_to, 3),
            "annualized": round(ann_to, 2),
            "est_cost_drag": round(cost_drag, 4),
            "net_cagr_10bps": (round(mc - cost_drag, 4) if mc is not None else None),
        }
    return res


def _vol_target(daily_chunks: list, st: dict) -> list:
    """Barroso-Santa-Clara vol-targeting on the strategy's own daily return stream.

    Scales each day's exposure by target_vol / yesterday's EWMA daily-vol forecast
    (no look-ahead), capped at vol_exposure_cap; the un-invested fraction earns the
    daily risk-free rate. Returns the vol-targeted return for each rebalance period
    (the daily series recompounded within each period window)."""
    lam = float(st.get("vol_ewma_lambda", 0.94))
    tgt = float(st.get("vol_target_annual", 0.27))
    cap = float(st.get("vol_exposure_cap", 1.0))
    rf_daily = config.RISK_FREE_ANNUAL / config.TRADING_DAYS
    full = np.concatenate([np.nan_to_num(c) for c in daily_chunks])
    lengths = [len(c) for c in daily_chunks]
    n = len(full)
    if n == 0:
        return [float(np.prod(1 + np.nan_to_num(c)) - 1.0) for c in daily_chunks]
    var = np.empty(n)
    var[0] = np.nanvar(full[:21]) if n > 21 else max(float(np.nanvar(full)), 1e-6)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * full[t - 1] ** 2
    fvol = np.sqrt(var * config.TRADING_DAYS)
    lev = np.minimum(np.where(fvol > 0, tgt / fvol, 1.0), cap)
    scaled = lev * full + (1.0 - lev) * rf_daily
    out, i = [], 0
    for ln in lengths:
        out.append(float(np.prod(1 + scaled[i:i + ln]) - 1.0))
        i += ln
    return out


def _bench_fwd(series: pd.Series, d: pd.Timestamp, hold_days: int) -> float:
    past = series[series.index <= d]
    fwd = series[series.index > d]
    if past.empty or len(fwd) < 1:
        return np.nan
    j = min(hold_days, len(fwd)) - 1
    return float(fwd.iloc[j] / past.iloc[-1] - 1.0)


def _summarize(ic_records, decile_rets, model_rets, spx_rets, ndx_rets, dates, ppy=12.0) -> dict:
    ics = pd.Series([r["ic"] for r in ic_records]) if ic_records else pd.Series(dtype=float)
    ic_mean = float(ics.mean()) if len(ics) else np.nan
    ic_std = float(ics.std()) if len(ics) else np.nan
    ic_ir = float(ic_mean / ic_std) if ic_std and ic_std > 0 else np.nan
    ic_t = float(ic_ir * np.sqrt(len(ics))) if len(ics) and not np.isnan(ic_ir) else np.nan
    hit = float((ics > 0).mean()) if len(ics) else np.nan

    decile_means = {str(k): (float(np.mean(v)) if v else None) for k, v in decile_rets.items()}
    # monotonicity = rank corr between decile index (0..9) and its mean fwd return
    dser = [decile_means[str(i)] for i in range(10)]
    if all(x is not None for x in dser):
        mono = float(spearmanr(np.arange(10), dser)[0])
        tb_spread = float(dser[9] - dser[0])
    else:
        mono, tb_spread = None, None

    # equity curves (compounded)
    def curve(rets):
        s = pd.Series(rets).fillna(0.0)
        return (1 + s).cumprod()
    mc, sc, nc = curve(model_rets), curve(spx_rets), curve(ndx_rets)

    def cagr(c, n):
        if len(c) < 2 or c.iloc[-1] <= 0:
            return None
        yrs = n / ppy
        return float(c.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else None

    def sharpe_of(rets):
        s = pd.Series(rets).dropna()
        if s.std() == 0 or len(s) < 4:
            return None
        return float(np.sqrt(ppy) * s.mean() / s.std())

    def maxdd(c):
        if len(c) < 2:
            return None
        return float((c / c.cummax() - 1).min())

    n_periods = len(dates)
    return {
        "ok": True,
        "n_rebalances": n_periods,
        "period": [str(dates[0].date()), str(dates[-1].date())] if dates else None,
        "ic": {"mean": ic_mean, "std": ic_std, "ir": ic_ir, "tstat": ic_t,
               "hit_rate": hit, "n": len(ics)},
        "ic_series": [{"date": str(r["date"].date()), "ic": r["ic"]} for r in ic_records],
        "deciles": decile_means,
        "monotonicity": mono,
        "tb_spread": tb_spread,
        "performance": {
            "model": {"cagr": cagr(mc, n_periods), "sharpe": sharpe_of(model_rets),
                      "max_drawdown": maxdd(mc), "total_return": float(mc.iloc[-1] - 1) if len(mc) else None},
            "sp500": {"cagr": cagr(sc, n_periods), "sharpe": sharpe_of(spx_rets),
                      "max_drawdown": maxdd(sc), "total_return": float(sc.iloc[-1] - 1) if len(sc) else None},
            "nasdaq": {"cagr": cagr(nc, n_periods), "sharpe": sharpe_of(ndx_rets),
                       "max_drawdown": maxdd(nc), "total_return": float(nc.iloc[-1] - 1) if len(nc) else None},
        },
        "curves": {
            "dates": [str(d.date()) for d in dates],
            "model": [round(float(x), 4) for x in mc],
            "sp500": [round(float(x), 4) for x in sc],
            "nasdaq": [round(float(x), 4) for x in nc],
        },
    }
