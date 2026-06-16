"""Scan momentum variants we haven't tried, incl. momentum ACCELERATION (the
change in momentum), to see which have IC/alpha at monthly & quarterly horizons."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

import config
from qmodel import engine

LB = 251
FILING_LAG = np.timedelta64(90, "D")
data = engine._load_bt_data(); bm = engine._benchmarks_cached()

FACTORS = ["mom_12_1", "mom_6_1", "mom_accel_q", "mom_accel_12", "mom_volscaled",
           "high_52w", "reversal_1m"]

prep = {}
for ck, blob in data.items():
    pr = blob.get("prices")
    if pr is None or len(pr) < 360: continue
    arr = np.asarray(pr.values, float)
    pidx = np.asarray(pr.index.values, "datetime64[ns]")
    rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:]/arr[:-1]-1
    fh = blob.get("fund_hist"); fidx = None; mcap = None
    if fh is not None and not fh.empty and "calculated_market_cap" in fh.columns:
        fidx = np.asarray(fh.index.values, "datetime64[ns]")
        mcap = np.asarray(fh["calculated_market_cap"].values, float)
    prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "fidx": fidx, "mcap": mcap}

def mcap_asof(P, dlag):
    if P["fidx"] is None: return np.nan
    p = int(np.searchsorted(P["fidx"], dlag, side="right"))-1
    return P["mcap"][p] if p >= 0 else np.nan

def run(hold_days):
    spx = bm["SP500"].dropna()
    monthly = pd.date_range(spx.index.min()+pd.Timedelta(days=500),
                            spx.index.max()-pd.Timedelta(days=hold_days+1), freq="ME")
    step = max(1, hold_days // 21)
    rebal = list(monthly[::step])
    ics = {f: [] for f in FACTORS}; dec_a = {f: np.zeros(10) for f in FACTORS}; dec_c = {f: np.zeros(10) for f in FACTORS}
    for d in rebal:
        dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt-FILING_LAG
        cand = []
        for ck, P in prep.items():
            pos = int(np.searchsorted(P["pidx"], dt, side="right"))-1
            if pos < 315 or (len(P["arr"])-1-pos) < hold_days: continue
            mc = mcap_asof(P, dlag)
            if mc is None or np.isnan(mc): continue
            cand.append((mc, ck, pos))
        if len(cand) < 50: continue
        cand.sort(key=lambda x: x[0], reverse=True); cand = cand[:config.BT_UNIVERSE_SIZE]
        rec = {f: [] for f in FACTORS}; fwd = []
        for mc, ck, pos in cand:
            P = prep[ck]; arr = P["arr"]; w = P["rets"][pos-LB+1:pos+1]; sd = w.std(ddof=1)
            mom12 = arr[pos-20]/arr[pos-251]-1
            mom12_prev = arr[pos-83]/arr[pos-314]-1
            vals = {
                "mom_12_1": mom12,
                "mom_6_1": arr[pos-20]/arr[pos-126]-1,
                "mom_accel_q": (arr[pos]/arr[pos-63]-1)-(arr[pos-63]/arr[pos-126]-1),
                "mom_accel_12": mom12-mom12_prev,
                "mom_volscaled": mom12/(np.sqrt(252)*sd) if sd > 0 else np.nan,
                "high_52w": arr[pos]/arr[pos-251:pos+1].max(),
                "reversal_1m": arr[pos]/arr[pos-20]-1,
            }
            for f in FACTORS: rec[f].append(vals[f])
            fwd.append(arr[pos+hold_days]/arr[pos]-1)
        fwd = np.array(fwd)
        for f in FACTORS:
            v = np.array(rec[f], float); m = ~np.isnan(v) & ~np.isnan(fwd)
            if m.sum() < 30 or np.unique(v[m]).size < 10: continue
            ic, _ = spearmanr(v[m], fwd[m])
            if not np.isnan(ic): ics[f].append(ic)
            try:
                dec = pd.qcut(pd.Series(v[m]).rank(method="first"), 10, labels=False).values
                fm = fwd[m]
                for i in range(10):
                    s = dec == i
                    if s.any(): dec_a[f][i] += fm[s].mean(); dec_c[f][i] += 1
            except ValueError: pass
    print(f"\n--- holding {hold_days//21}M · {len(rebal)} periods ---")
    print(f"{'factor':14} {'IC':>8} {'t':>6} {'hit':>5} {'mono':>6}")
    out = []
    for f in FACTORS:
        s = pd.Series(ics[f])
        if len(s) < 5: continue
        icm = s.mean(); t = icm/s.std()*np.sqrt(len(s)) if s.std() > 0 else 0
        hit = (np.sign(s) == np.sign(icm)).mean()
        dm = np.divide(dec_a[f], dec_c[f], out=np.full(10, np.nan), where=dec_c[f] > 0)
        mono = spearmanr(np.arange(10), dm)[0] if np.isfinite(dm).all() else np.nan
        out.append((abs(icm), f, icm, t, hit, mono))
    for _, f, icm, t, hit, mono in sorted(out, reverse=True):
        print(f"{f:14} {icm:>+8.4f} {t:>+6.2f} {hit:>5.0%} {mono:>+6.2f}")

run(21); run(63)
