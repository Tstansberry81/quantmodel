"""Evaluate configs across trailing lookback windows vs BOTH indexes, plus
drawdown. Tests a beta spectrum from defensive (current) to high-CAGR."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config
from qmodel import engine, scoring, backtest as bt
from qmodel.equations import FACTORS, default_params

N = config.N_STOCKS; HOLD = 21; LB = 251; SKIP = 21
FILING_LAG = np.timedelta64(bt.FILING_LAG_DAYS, "D")
data = engine._load_bt_data(); bm = engine._benchmarks_cached()
mkt = bm["SP500"].dropna(); mret_full = mkt.pct_change()
ndx = bm["NASDAQ"].dropna()
fund_ids = {f: FACTORS[f]["ratio_id"] for f in FACTORS if FACTORS[f]["ratio_id"]}
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
        for fn, rid in fund_ids.items(): fcols[fn] = np.asarray(fh[rid].values, float) if rid in fh.columns else None
        if "calculated_market_cap" in fh.columns: mcap = np.asarray(fh["calculated_market_cap"].values, float)
    prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret, "fidx": fidx, "fcols": fcols, "mcap": mcap,
                "sector": blob.get("meta", {}).get("sector", "Unknown")}
def asof(fidx, varr, dlag):
    if fidx is None or varr is None: return np.nan
    p = int(np.searchsorted(fidx, dlag, side="right")) - 1
    return varr[p] if p >= 0 else np.nan
def bench_fwd(series, d):
    past = series[series.index <= d]; fwd = series[series.index > d]
    if past.empty or len(fwd) < 1: return np.nan
    return float(fwd.iloc[min(HOLD, len(fwd))-1]/past.iloc[-1]-1)
params = default_params(); FACT = list(FACTORS.keys())
spx = mkt; monthly = spx.index[spx.index >= spx.index.min()+pd.Timedelta(days=400)]
rebal = list(monthly[:-HOLD][::HOLD])
Zs, fwds, bdates, spxf, ndxf, revg = [], [], [], [], [], []
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
    rows = []
    for mc, ck, pos in cand:
        P = prep[ck]; arr = P["arr"]; w = P["rets"][pos-LB+1:pos+1]; wm = P["mret"][pos-LB+1:pos+1]
        sd = w.std(ddof=1); neg = w[w < 0]; dd = neg.std(ddof=1) if neg.size > 1 else np.nan
        p6, p12 = arr[pos-125], arr[pos-251]; hi = arr[pos-251:pos+1].max()
        ok = ~np.isnan(w) & ~np.isnan(wm); rm = np.nan
        if ok.sum() > 60:
            rr, mm = w[ok], wm[ok]
            if mm.var() > 0:
                beta = np.cov(rr, mm)[0,1]/mm.var(); resid = rr-(rr.mean()-beta*mm.mean())-beta*mm
                rv = resid.std(ddof=1)
                if rv > 0 and resid.size > SKIP: rm = resid[:-SKIP].sum()/rv
        row = {"company_key": ck, "sector": P["sector"], "pit_mcap": mc,
               "momentum": arr[pos-20]/arr[pos-251]-1, "resid_mom": rm,
               "mom_accel": ((arr[pos]/p6-1)-(p6/p12-1)) if p6>0 and p12>0 else np.nan,
               "mom_52w_high": arr[pos]/hi if hi>0 else np.nan,
               "sharpe": (np.sqrt(252)*(w.mean()-config.RISK_FREE_ANNUAL/252)/sd) if sd>0 else np.nan,
               "sortino": (np.sqrt(252)*(w.mean()-config.RISK_FREE_ANNUAL/252)/dd) if dd and dd>0 else np.nan,
               "volatility": np.sqrt(252)*sd, "fwd_ret": arr[pos+HOLD]/arr[pos]-1}
        for fn in fund_ids: row[fn] = asof(P["fidx"], P["fcols"].get(fn), dlag)
        rows.append(row)
    cs = scoring.compute_scores(pd.DataFrame(rows), params); valid = cs.dropna(subset=["fwd_ret"])
    zc = [f"z_{f}" for f in FACT if f"z_{f}" in valid.columns]
    Zs.append(valid[zc].to_numpy(float)); fwds.append(valid["fwd_ret"].to_numpy(float))
    revg.append(valid["rev_growth"].to_numpy(float))
    bdates.append(d); spxf.append(bench_fwd(spx, d)); ndxf.append(bench_fwd(ndx, d))
fnames = [c[2:] for c in zc]; fi = {f: i for i, f in enumerate(fnames)}
spxf = np.nan_to_num(np.array(spxf)); ndxf = np.nan_to_num(np.array(ndxf))
def vec(d):
    w = np.zeros(len(fnames))
    for k, v in d.items(): w[fi[k]] = v
    return w
def cagr_of(r, ppy=12):
    eq = np.cumprod(1+np.nan_to_num(r)); return eq[-1]**(ppy/len(r))-1 if eq[-1] > 0 else -1
def maxdd_of(r):
    eq = np.cumprod(1+np.nan_to_num(r)); return (eq/np.maximum.accumulate(eq)-1).min()
def model_rets(w, min_g=None):
    """Top-N by composite, optionally restricted to rev_growth >= min_g (hard filter).
    Also returns the avg number of names passing the filter per rebalance."""
    out, npass = [], []
    for Z, fwd, g in zip(Zs, fwds, revg):
        comp = Z @ w
        if min_g is not None:
            ok = np.where((g >= min_g) & ~np.isnan(g))[0]
            npass.append(len(ok))
            if len(ok) < 3: out.append(0.0); continue
            sub = comp[ok]; sel = ok[np.argpartition(-sub, min(N, len(sub)-1))[:N]]
        else:
            sel = np.argpartition(-comp, min(N, len(comp)-1))[:N]
        out.append(np.nanmean(fwd[sel]))
    return np.array(out), (np.mean(npass) if npass else len(Zs[0]))
base = vec({"rev_growth":1.5,"momentum":1.0,"sharpe":1.25,"mom_accel":1.5,"size":0.5})
wins = {"2Y":24,"5Y":60,"10Y":120,"20Y":240,"MAX":len(Zs)}
print("Rev-growth HARD FILTER on the current model. CAGR (model% / +aSP / +aNDX), maxDD, avg #names passing\n")
print("filter".ljust(16) + "".join(f"{w:>22}" for w in wins) + "   maxDD  #pass")
for name, mg in [("none (current)", None), (">=10% YoY", 0.10), (">=15% YoY", 0.15),
                 (">=20% YoY", 0.20), (">=25% YoY", 0.25)]:
    r, npass = model_rets(base, mg); line = name.ljust(16)
    for win, k in wins.items():
        k = min(k, len(r)); mr = cagr_of(r[-k:]); sp = cagr_of(spxf[-k:]); nd = cagr_of(ndxf[-k:])
        line += f"{mr*100:5.1f}/{(mr-sp)*100:+4.1f}/{(mr-nd)*100:+4.1f}".rjust(22)
    line += f"   {maxdd_of(r)*100:4.0f}%  {npass:.0f}"
    print(line)
