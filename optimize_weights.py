"""Factor-weight search. Precompute the z-scored factor panel once, then
evaluate many weight vectors fast (composite = panel @ weights). Objective:
improve Sharpe / Calmar (CAGR/|maxDD|) vs the current weights — i.e. tame
drawdown without giving up CAGR. Validates winners on full + recent-10Y.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import config
from qmodel import engine, scoring, backtest as bt
from qmodel.equations import FACTORS, default_params

N = config.N_STOCKS
HOLD = 21; LB = 251; SKIP = 21
FILING_LAG = np.timedelta64(bt.FILING_LAG_DAYS, "D")
data = engine._load_bt_data(); bm = engine._benchmarks_cached()
mkt_ret = bm["SP500"].dropna().pct_change()
fund_ids = {f: FACTORS[f]["ratio_id"] for f in FACTORS if FACTORS[f]["ratio_id"]}

# ---- precompute per-name arrays ----
prep = {}
for ck, blob in data.items():
    pr = blob.get("prices")
    if pr is None or len(pr) < LB + HOLD + 2: continue
    arr = np.asarray(pr.values, float); pidx = np.asarray(pr.index.values, "datetime64[ns]")
    rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:]/arr[:-1]-1
    mret = np.asarray(mkt_ret.reindex(pr.index, method="ffill").values, float)
    fh = blob.get("fund_hist"); fidx = None; fcols = {}; mcap = None
    if fh is not None and not fh.empty:
        fidx = np.asarray(fh.index.values, "datetime64[ns]")
        for fn, rid in fund_ids.items():
            fcols[fn] = np.asarray(fh[rid].values, float) if rid in fh.columns else None
        if "calculated_market_cap" in fh.columns: mcap = np.asarray(fh["calculated_market_cap"].values, float)
    prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret, "fidx": fidx, "fcols": fcols, "mcap": mcap,
                "sector": blob.get("meta", {}).get("sector", "Unknown")}

def asof(fidx, varr, dlag):
    if fidx is None or varr is None: return np.nan
    p = int(np.searchsorted(fidx, dlag, side="right")) - 1
    return varr[p] if p >= 0 else np.nan

params = default_params()
FACT = list(FACTORS.keys())

def build_panel(step):
    spx = bm["SP500"].dropna(); monthly = spx.index[spx.index >= spx.index.min()+pd.Timedelta(days=400)]
    rebal = list(monthly[:-HOLD][::step])
    Zs, fwds, bdates = [], [], []
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
                   "volatility": np.sqrt(252)*sd,
                   "fwd_ret": arr[pos+HOLD]/arr[pos]-1}
            for fn in fund_ids: row[fn] = asof(P["fidx"], P["fcols"].get(fn), dlag)
            rows.append(row)
        cs = scoring.compute_scores(pd.DataFrame(rows), params)
        valid = cs.dropna(subset=["fwd_ret"])
        zcols = [f"z_{f}" for f in FACT if f"z_{f}" in valid.columns]
        Zs.append(valid[zcols].to_numpy(float)); fwds.append(valid["fwd_ret"].to_numpy(float)); bdates.append(d)
    return Zs, fwds, bdates, [c[2:] for c in zcols]

def metrics(Zs, fwds, w, ppy=12):
    r = []
    for Z, fwd in zip(Zs, fwds):
        comp = Z @ w
        idx = np.argpartition(-comp, min(N, len(comp)-1))[:N]
        r.append(np.nanmean(fwd[idx]))
    r = np.array(r); eq = np.cumprod(1+np.nan_to_num(r))
    yrs = len(r)/ppy; cagr = eq[-1]**(1/yrs)-1 if eq[-1] > 0 else -1
    sharpe = np.sqrt(ppy)*r.mean()/r.std() if r.std() > 0 else 0
    dd = (eq/np.maximum.accumulate(eq)-1).min()
    return cagr, sharpe, dd, (cagr/abs(dd) if dd < 0 else np.nan)

print("Building factor panel (monthly, MAX)...")
Zs, fwds, bdates, fnames = build_panel(HOLD)   # stride 21 trading days = monthly
print(f"  {len(Zs)} rebalances, factors: {fnames}\n")
fi = {f: i for i, f in enumerate(fnames)}
def vec(d):
    w = np.zeros(len(fnames))
    for k, v in d.items(): w[fi[k]] = v
    return w

cur = vec({"momentum":1.5,"mom_accel":1.0,"mom_52w_high":1.0,"rev_growth":1.0,"roic":1.0,"fcf_yield":0.5})

# ---- random search ----
rng = np.random.default_rng(7)
pool = ["momentum","mom_accel","mom_52w_high","resid_mom","roic","rev_growth",
        "fcf_yield","shareholder_yield","size","volatility","sharpe","sortino"]
pidx = [fi[p] for p in pool]
best = {"sharpe": (None,-9), "calmar": (None,-9), "cagr": (None,-9)}
results = []
for _ in range(4000):
    w = np.zeros(len(fnames))
    mask = rng.random(len(pool)) < 0.5
    w[np.array(pidx)[mask]] = rng.uniform(0.25, 2.0, mask.sum())
    if w.sum() == 0: continue
    c, s, dd, cal = metrics(Zs, fwds, w)
    results.append((w, c, s, dd, cal))
    if s > best["sharpe"][1]: best["sharpe"] = (w, s)
    if not np.isnan(cal) and cal > best["calmar"][1]: best["calmar"] = (w, cal)
    if c > best["cagr"][1]: best["cagr"] = (w, c)

def show(tag, w):
    c, s, dd, cal = metrics(Zs, fwds, w)
    nz = {fnames[i]: round(float(w[i]),2) for i in range(len(w)) if w[i] > 0}
    print(f"{tag:22} CAGR={c*100:5.1f}% Sharpe={s:.2f} maxDD={dd*100:5.0f}% Calmar={cal:.2f}  {nz}")

show("current default", cur)
print("\n=== best by objective (random search, 4000 samples) ===")
show("max Sharpe", best["sharpe"][0])
show("max Calmar (DD-adj)", best["calmar"][0])
show("max CAGR", best["cagr"][0])

# ---- robustness: split-sample validation (in-sample search was full period) ----
bts = pd.to_datetime(bdates)
cut = bts.max() - pd.DateOffset(years=10)
m_recent = np.asarray(bts >= cut); m_early = ~m_recent
def sub(mask): return [Zs[i] for i in range(len(Zs)) if mask[i]], [fwds[i] for i in range(len(fwds)) if mask[i]]
Zr, Fr = sub(m_recent); Ze, Fe = sub(m_early)
print("\n=== robustness: same weights, different sub-periods (CAGR / Sharpe / maxDD) ===")
cands = {"current": cur, "max-Sharpe": best["sharpe"][0], "max-Calmar": best["calmar"][0], "max-CAGR": best["cagr"][0]}
print(f"{'config':12} {'EARLY (2005-16)':>26}   {'RECENT (2016-26)':>26}")
for name, w in cands.items():
    ce, se, de, _ = metrics(Ze, Fe, w); cr, sr, dr, _ = metrics(Zr, Fr, w)
    print(f"{name:12} {ce*100:6.1f}% {se:5.2f} {de*100:5.0f}%     {cr*100:6.1f}% {sr:5.2f} {dr*100:5.0f}%")
