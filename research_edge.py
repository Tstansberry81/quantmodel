"""Edge honesty metrics + a light revenue-growth-gate experiment.

Part 1: evaluate the full-spec Edge against a RICHER metric set than CAGR/Sharpe
(Sortino, Calmar, information ratio, up/down capture, rolling-window win-rate,
beta, skew, tails) -- a tougher honesty lens.

Part 2: the Edge is market-data-only by design (fundamentals hurt the short-term
model in prior tests). Test whether a LIGHT YoY revenue-growth gate (>=5%, "for
growth purposes") helps or hurts -- thresholds 0/5/10/15% vs no gate. The gate is
applied point-in-time (90-day filing lag, no lookahead).

Reads cached data via edge_lib + qmodel; does not touch the live model.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from qmodel import engine

HOLD, N, FLOOR, CAP, REG, COST = 42, 20, 2e9, 0.50, 0.25, 10.0
W = {"accel": 1.0}
LAG = np.timedelta64(90, "D")

print("Building Edge panel ...")
pan = E.load_edge_panel(hold=HOLD)
ppy = pan.ppy
data = engine._load_bt_data()

# point-in-time YoY revenue growth lookup (growth_revenue_1y), 90-day filing lag
rg = {}
for ck, blob in data.items():
    fh = blob.get("fund_hist")
    if fh is not None and not fh.empty and "growth_revenue_1y" in fh.columns:
        rg[ck] = (np.asarray(fh.index.values, "datetime64[ns]"),
                  np.asarray(fh["growth_revenue_1y"].values, float))


def rg_asof(ck, d):
    t = rg.get(ck)
    if not t:
        return np.nan
    idx, arr = t
    p = int(np.searchsorted(idx, np.datetime64(pd.Timestamp(d), "ns") - LAG, "right")) - 1
    return arr[p] if p >= 0 else np.nan


def edge_returns(rev_thr=None):
    """Full-spec Edge per-period NET returns, optionally gating candidates to
    YoY revenue growth >= rev_thr. Returns (net_rets, avg_universe_after_gate)."""
    holds, turn, prev, npass = [], [], None, []
    for i, df in enumerate(pan.panels):
        d = pan.bdates[i]
        sub = df[df["pit_mcap"] >= FLOOR].copy()
        if rev_thr is not None:
            sub["_rg"] = [rg_asof(ck, d) for ck in sub["company_key"]]
            sub = sub[sub["_rg"] >= rev_thr]
        npass.append(len(sub))
        cks = E.corr_cap_select(sub, asof=d, n=N, weights=W, cap=CAP) if len(sub) >= 3 else []
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    r = E.period_returns(pan, holds)                                   # equal-wt top-N fwd
    r = np.where(pan.ma200_on, r, REG * r + (1 - REG) * pan.rf_per)    # 200dMA regime
    r = r - (COST / 1e4) * np.array(turn)                             # trading costs
    return r, float(np.mean(npass)), float(np.mean(turn)) * ppy


def cagr(r):
    eq = np.cumprod(1 + np.nan_to_num(r)); return eq[-1] ** (ppy / len(r)) - 1 if eq[-1] > 0 else -1
def maxdd(r):
    eq = np.cumprod(1 + np.nan_to_num(r)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def metrics(r, bench):
    r = np.asarray(r, float); b = np.asarray(bench, float)
    ex = r - pan.rf_per
    sd = np.nanstd(r); dn = r[r < pan.rf_per]
    downside = np.nanstd(dn - pan.rf_per) if len(dn) > 1 else np.nan
    c = cagr(r); dd = maxdd(r)
    sharpe = np.sqrt(ppy) * np.nanmean(ex) / sd if sd > 0 else np.nan
    sortino = np.sqrt(ppy) * np.nanmean(ex) / downside if downside and downside > 0 else np.nan
    calmar = c / abs(dd) if dd < 0 else np.nan
    diff = r - b
    info = np.sqrt(ppy) * np.nanmean(diff) / np.nanstd(diff) if np.nanstd(diff) > 0 else np.nan
    beta = np.cov(r, b)[0, 1] / np.var(b) if np.var(b) > 0 else np.nan
    up = b > 0; dnm = b < 0
    up_cap = np.nanmean(r[up]) / np.nanmean(b[up]) if up.any() and np.nanmean(b[up]) != 0 else np.nan
    dn_cap = np.nanmean(r[dnm]) / np.nanmean(b[dnm]) if dnm.any() and np.nanmean(b[dnm]) != 0 else np.nan
    hit_bench = float((diff > 0).mean())
    # rolling 1-year (ppy-period) win rate vs the S&P
    k = int(round(ppy)); wins = 0; tot = 0
    for t in range(len(r) - k + 1):
        tot += 1
        if np.prod(1 + np.nan_to_num(r[t:t + k])) > np.prod(1 + np.nan_to_num(b[t:t + k])):
            wins += 1
    roll_win = wins / tot if tot else np.nan
    return {"CAGR": c, "Sharpe": sharpe, "Sortino": sortino, "maxDD": dd, "Calmar": calmar,
            "InfoRatio": info, "beta": beta, "up_capture": up_cap, "down_capture": dn_cap,
            "hit_vs_SP": hit_bench, "roll1Y_win": roll_win,
            "skew": float(pd.Series(r).skew()), "worst": float(np.min(r)), "best": float(np.max(r))}


# ============================ Part 1: honesty metrics =========================
base, _, base_to = edge_returns(None)
spx = pan.spxf
m = metrics(base, spx); ms = metrics(spx, spx)
print("\n" + "=" * 70)
print("1. EDGE HONESTY METRICS (full spec, net of costs, vs S&P)")
print("=" * 70)
print(f"  {'metric':<14}{'Edge':>12}{'S&P 500':>12}")
for key, lab in [("CAGR", "CAGR"), ("Sharpe", "Sharpe"), ("Sortino", "Sortino"),
                 ("maxDD", "max drawdown"), ("Calmar", "Calmar"), ("InfoRatio", "info ratio"),
                 ("beta", "beta vs S&P"), ("up_capture", "up capture"), ("down_capture", "down capture"),
                 ("hit_vs_SP", "hit-rate vs S&P"), ("roll1Y_win", "rolling-1Y win"),
                 ("skew", "skew"), ("worst", "worst period"), ("best", "best period")]:
    def fmt(v):
        if key in ("CAGR", "maxDD", "up_capture", "down_capture", "hit_vs_SP", "roll1Y_win", "worst", "best"):
            return f"{v*100:.1f}%" if not np.isnan(v) else "—"
        return f"{v:.2f}" if not np.isnan(v) else "—"
    print(f"  {lab:<14}{fmt(m[key]):>12}{fmt(ms[key]):>12}")
print(f"  annual turnover: {base_to:.1f}x")
print("  Read: Calmar (CAGR/maxDD), info ratio (excess/tracking-error), and down-capture")
print("  <1 (cushions losses) are the honest 'is it more than beta?' checks.")

# ===================== Part 2: revenue-growth gate test =======================
print("\n" + "=" * 70)
print("2. LIGHT YoY REVENUE-GROWTH GATE (>= threshold), full Edge spec")
print("=" * 70)
print(f"  {'gate':<14}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}{'turn':>7}{'avg univ':>9}")
rows = [("none (mkt-only)", None), (">=0%", 0.0), (">=5%", 0.05), (">=10%", 0.10), (">=15%", 0.15)]
results = {}
for lab, thr in rows:
    r, univ, to = edge_returns(thr)
    mm = metrics(r, spx); results[lab] = r
    print(f"  {lab:<14}{mm['CAGR']*100:7.1f}%{mm['Sharpe']:8.2f}{mm['Sortino']:9.2f}"
          f"{mm['maxDD']*100:7.0f}%{to:6.1f}x{univ:9.0f}")
# window view for the 5% gate vs baseline
print("\n  Window view -- none vs >=5% gate (CAGR / Sharpe / maxDD):")
for lab in ("none (mkt-only)", ">=5%"):
    r = results[lab]
    line = f"  {lab:<14}"
    for wn, yr in (("1Y", 1), ("2Y", 2), ("5Y", 5), ("MAX", None)):
        k = len(r) if yr is None else min(int(round(yr * ppy)), len(r))
        rr = r[-k:]
        line += f" {wn}:{cagr(rr)*100:.0f}%/{(np.sqrt(ppy)*np.nanmean(rr-pan.rf_per)/np.nanstd(rr)):.2f}/{maxdd(rr)*100:.0f}%"
    print(line)
print("\nDone.")
