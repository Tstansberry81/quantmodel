"""EXPERIMENT: run the CHT signal (cheap = FCF-yield, uptrend = 52-week range
position; Hated dropped) on the EDGE's broad point-in-time universe instead of
CHT+'s ~100 hand-picked momentum names — to see if the signal actually travels.

Universe: top-1000 by point-in-time market cap each rebalance (a Russell-1000
proxy; true R1000 membership is the paused Norgate piece). Fundamentals lagged
90 days. 1-month (21 trading-day) rebalance, top-10 equal weight, vs S&P 500.

Run:  .venv/Scripts/python.exe cht_on_edge.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd

import config
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)

HOLD = 21            # 1-month rebalance
LB = 252             # 52-week lookback
TOPN = 10
UNIV = 1000          # top-N by PIT mcap
LAG = np.timedelta64(90, "D")
PPY = 252.0 / HOLD


def _prep():
    data = engine._load_bt_data()
    prep = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB + HOLD + 2:
            continue
        arr = np.asarray(pr.values, float)
        pidx = np.asarray(pr.index.values, "datetime64[ns]")
        s = pd.Series(arr)
        roll_hi = s.rolling(LB).max().values
        roll_lo = s.rolling(LB).min().values
        fh = blob.get("fund_hist"); fidx = mcap = fcfy = None
        if fh is not None and not fh.empty:
            fidx = np.asarray(fh.index.values, "datetime64[ns]")
            if "calculated_market_cap" in fh.columns:
                mcap = np.asarray(fh["calculated_market_cap"].values, float)
            if "ratio_fcf_yield" in fh.columns:
                fcfy = np.asarray(fh["ratio_fcf_yield"].values, float)
        prep[ck] = {"arr": arr, "pidx": pidx, "hi": roll_hi, "lo": roll_lo,
                    "fidx": fidx, "mcap": mcap, "fcfy": fcfy}
    return prep


def _asof(fidx, varr, dlag):
    if fidx is None or varr is None:
        return None
    p = int(np.searchsorted(fidx, dlag, side="right")) - 1
    if p < 0:
        return None
    v = varr[p]
    return None if (v != v) else float(v)


def run():
    prep = _prep()
    bm = engine._benchmarks_cached()
    spx = bm["SP500"].dropna()
    spx_idx = spx.index.values.astype("datetime64[ns]")
    cal = spx.index
    rebal = list(cal[cal >= cal.min() + pd.Timedelta(days=400)][:-HOLD][::HOLD])

    period_ret, sp_ret, dates, n_cand, n_fcf = [], [], [], [], []
    for d in rebal:
        dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt - LAG
        cands = []
        for ck, P in prep.items():
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB or (len(P["arr"]) - 1 - pos) < HOLD:
                continue
            mc = _asof(P["fidx"], P["mcap"], dlag)
            if mc is None:
                continue
            cands.append((mc, ck, pos))
        if len(cands) < 50:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)
        cands = cands[:UNIV]

        scored = []; fcf_have = 0
        for mc, ck, pos in cands:
            P = prep[ck]; price = P["arr"][pos]
            hi, lo = P["hi"][pos], P["lo"][pos]
            if not (hi > lo) or price <= 0:
                continue
            uptrend = max(0.0, min(100.0, (price - lo) / (hi - lo) * 100))
            fcfy = _asof(P["fidx"], P["fcfy"], dlag)
            if fcfy is not None:
                cheap = max(0.0, min(100.0, 50 + 50 * math.tanh(fcfy / 0.08))); fcf_have += 1
            else:
                cheap = 50.0
            score = 0.5 * cheap + 0.5 * uptrend
            fwd = P["arr"][pos + HOLD] / price - 1
            scored.append((score, fwd))
        if len(scored) < TOPN:
            continue
        # S&P forward return over the same hold window; skip the rebalance if
        # the S&P can't cover the full window (stocks already require it), so
        # model vs benchmark are always measured over identical spans.
        sp_pos = int(np.searchsorted(spx_idx, dt, side="right")) - 1
        if sp_pos < 0 or sp_pos + HOLD >= len(spx):
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:TOPN]
        period_ret.append(sum(f for _, f in picks) / len(picks))
        sp_ret.append(float(spx.iloc[sp_pos + HOLD] / spx.iloc[sp_pos] - 1))
        dates.append(pd.Timestamp(d)); n_cand.append(len(cands)); n_fcf.append(fcf_have)

    return (np.array(period_ret), np.array(sp_ret), pd.DatetimeIndex(dates),
            np.array(n_cand), np.array(n_fcf))


def perf(r):
    r = np.nan_to_num(np.asarray(r, float))
    if len(r) == 0:
        return 0.0, 0.0, 0.0, 0.0
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (PPY / len(r)) - 1 if eq[-1] > 0 else -1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    tot = float(eq[-1] - 1)
    win = float((r > 0).mean())
    return float(cagr), tot, dd, win


def main():
    pr, sr, dts, ncand, nfcf = run()
    T = len(pr)
    print(f"CHT signal (cheap=FCF-yield .50 / uptrend=52w-range .50) on the EDGE PIT top-{UNIV} "
          f"universe\n  top-{TOPN} equal-weight · 1-month rebalance · vs S&P 500\n"
          f"  {T} rebalances {dts[0].date()}-{dts[-1].date()} · "
          f"avg {ncand.mean():.0f} names/rebalance · FCF-yield coverage {100*nfcf.sum()/ncand.sum():.0f}%\n")
    hdr = f"{'window':>7}{'CAGR':>9}{'totRet':>10}{'maxDD':>9}{'winRate':>9}{'S&P CAGR':>10}{'excess':>9}{'rebal':>7}"
    print(hdr); print("-" * len(hdr))
    for years in (1, 2, 5, 10, 20):
        k = min(int(round(years * PPY)), T)
        c, tot, dd, win = perf(pr[T - k:])
        sc, _, _, _ = perf(sr[T - k:])
        print(f"{str(years)+'y':>7}{c*100:>8.1f}%{tot*100:>9.0f}%{dd*100:>8.0f}%"
              f"{win*100:>8.0f}%{sc*100:>9.1f}%{(c-sc)*100:>+8.1f}%{k:>7}")
    print("\nNo $ liquidity floor or costs; gross, equal-weight. Universe is top-1000 by PIT")
    print("mcap (broad, ~all >= a few $B) - far less survivorship-curated than CHT+'s 100 names,")
    print("but still missing most delisted/dead names (absolute returns remain somewhat high).")


if __name__ == "__main__":
    main()
