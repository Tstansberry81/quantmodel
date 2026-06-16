"""Scan candidate KPIs we haven't weighted yet: value (earnings/FCF yield, EV/EBITDA),
quality (gross margin, ROE, net margin), growth (EPS growth), leverage, plus three
computed from prices: idiosyncratic vol (low-IVOL anomaly), market beta (low-beta),
and 1-month reversal. Reports raw IC (sign=direction), t-stat, and decile monotonicity.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import config
from qmodel import engine, backtest as bt
from qmodel.equations import FACTORS

HOLD = 21; LB = 251; SKIP = 21
FILING_LAG = np.timedelta64(bt.FILING_LAG_DAYS, "D")
data = engine._load_bt_data(); bm = engine._benchmarks_cached()
mret_full = bm["SP500"].dropna().pct_change()
# fundamental ratioIds already in fund_hist
FUND = {"earnings_yield":"ratio_earnings_yield","ev_ebitda":"ratio_ev_to_ebitda",
        "gross_margin":"ratio_gross_profit_margin","roe":"ratio_return_on_equity",
        "net_margin":"ratio_net_profit_margin","eps_growth":"growth_diluted_eps_1y",
        "debt_to_equity":"ratio_debt_to_equity","fcf_yield":"ratio_fcf_yield"}
PRICE = ["ivol","beta","reversal_1m"]
ALL = list(FUND) + PRICE
prep = {}
for ck, blob in data.items():
    pr = blob.get("prices")
    if pr is None or len(pr) < LB + HOLD + 2: continue
    arr = np.asarray(pr.values, float); pidx = np.asarray(pr.index.values, "datetime64[ns]")
    rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:]/arr[:-1]-1
    mret = np.asarray(mret_full.reindex(pr.index, method="ffill").values, float)
    fh = blob.get("fund_hist"); fidx = None; fcols = {}; mcap = None
    if fh is not None and not fh.empty:
        fidx = np.asarray(fh.index.values, "datetime64[ns]")
        for nm, rid in {**FUND, "mcap":"calculated_market_cap"}.items():
            fcols[nm] = np.asarray(fh[rid].values, float) if rid in fh.columns else None
        mcap = fcols.get("mcap")
    prep[ck] = {"arr":arr,"pidx":pidx,"rets":rets,"mret":mret,"fidx":fidx,"fcols":fcols,"mcap":mcap}
def asof(fidx, varr, dlag):
    if fidx is None or varr is None: return np.nan
    p = int(np.searchsorted(fidx, dlag, side="right")) - 1
    return varr[p] if p >= 0 else np.nan
mkt = bm["SP500"].dropna(); monthly = mkt.index[mkt.index >= mkt.index.min()+pd.Timedelta(days=400)]
rebal = list(monthly[:-HOLD][::HOLD])
ics = {f: [] for f in ALL}; dacc = {f: np.zeros(10) for f in ALL}; dcnt = {f: np.zeros(10) for f in ALL}
for d in rebal:
    dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt - FILING_LAG
    cand = []
    for ck, P in prep.items():
        pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
        if pos < LB or (len(P["arr"])-1-pos) < HOLD: continue
        mc = asof(P["fidx"], P["mcap"], dlag)
        if mc is None or np.isnan(mc): continue
        cand.append((mc, ck, pos))
    if len(cand) < 50: continue
    cand.sort(key=lambda x: x[0], reverse=True); cand = cand[:config.BT_UNIVERSE_SIZE]
    rec = {f: [] for f in ALL}; fwd = []
    for mc, ck, pos in cand:
        P = prep[ck]; arr = P["arr"]; w = P["rets"][pos-LB+1:pos+1]; wm = P["mret"][pos-LB+1:pos+1]
        ok = ~np.isnan(w) & ~np.isnan(wm); ivol = beta = np.nan
        if ok.sum() > 60:
            rr, mm = w[ok], wm[ok]
            if mm.var() > 0:
                beta = np.cov(rr, mm)[0,1]/mm.var(); resid = rr-(rr.mean()-beta*mm.mean())-beta*mm
                ivol = resid.std(ddof=1)
        for nm in FUND: rec[nm].append(asof(P["fidx"], P["fcols"].get(nm), dlag))
        rec["ivol"].append(ivol); rec["beta"].append(beta); rec["reversal_1m"].append(arr[pos]/arr[pos-20]-1)
        fwd.append(arr[pos+HOLD]/arr[pos]-1)
    fwd = np.array(fwd)
    for f in ALL:
        v = np.array(rec[f], float); m = ~np.isnan(v) & ~np.isnan(fwd)
        if m.sum() < 30 or np.unique(v[m]).size < 10: continue
        ic, _ = spearmanr(v[m], fwd[m])
        if not np.isnan(ic): ics[f].append(ic)
        try:
            dec = pd.qcut(pd.Series(v[m]).rank(method="first"), 10, labels=False).values
            fm = fwd[m]
            for i in range(10):
                sel = dec == i
                if sel.any(): dacc[f][i] += fm[sel].mean(); dcnt[f][i] += 1
        except ValueError: pass
print(f"Monthly horizon, {len(rebal)} rebalances, PIT top-{config.BT_UNIVERSE_SIZE}")
print("(positive IC = higher value -> higher return; negative = lower value better)\n")
print(f"{'KPI':16} {'IC_mean':>8} {'IC_t':>6} {'mono':>6}  interpretation")
rows = []
for f in ALL:
    s = pd.Series(ics[f])
    if len(s) < 5: continue
    icm = s.mean(); ict = icm/s.std()*np.sqrt(len(s)) if s.std() > 0 else 0
    dm = np.divide(dacc[f], dcnt[f], out=np.full(10, np.nan), where=dcnt[f] > 0)
    mono = spearmanr(np.arange(10), dm)[0] if np.isfinite(dm).all() else np.nan
    rows.append((abs(icm), f, icm, ict, mono))
for _, f, icm, ict, mono in sorted(rows, reverse=True):
    tag = "STRONG" if abs(ict) > 2 else ("weak" if abs(ict) > 1 else "noise")
    print(f"{f:16} {icm:>+8.4f} {ict:>+6.2f} {mono:>+6.2f}  {tag}")
