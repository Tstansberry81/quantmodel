"""Is the Edge real downturn alpha, or a fair-weather momentum bet leaning on the
200dMA regime switch? Conditional performance, skew/crash profile, crisis years,
and a regime-on/off decomposition. Reads edge_lib only; does not modify it."""
from __future__ import annotations
import numpy as np, pandas as pd
import edge_lib as E

SIG = tuple(sorted({"accel": 1.0}.items()))
RF = E.RF_PER  # per-42d-period risk-free (note: E.RF_PER uses module PPY=12; we
               # only use it for cash-substitution context, not annualization)

# hold=42 -> ~6 periods/yr. Derive the true PPY from actual rebalance spacing so
# Sharpe annualization is correct (module-level E.PPY=12 is for the hold=21 default).
def true_ppy(bdates):
    gap = np.median(np.diff(bdates.values).astype("timedelta64[D]").astype(int))
    return 365.25 / gap

def ann_sharpe(r, ppy):
    r = np.asarray(r, float)
    sd = np.nanstd(r)
    return float(np.sqrt(ppy) * np.nanmean(r) / sd) if sd > 0 else 0.0

def cagr(r, ppy):
    r = np.nan_to_num(np.asarray(r, float))
    eq = np.cumprod(1 + r)
    return eq[-1] ** (ppy / len(r)) - 1 if len(r) and eq[-1] > 0 else -1.0

def maxdd(r):
    r = np.nan_to_num(np.asarray(r, float))
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(r) else 0.0

def skew(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    m, s = r.mean(), r.std()
    return float(((r - m) ** 3).mean() / s ** 3) if s > 0 else 0.0

# ---------------------------------------------------------------------------
# Pull the full-history Edge series. regime_expo=0.25 = full spec (regime ON);
# regime_expo=1.0 = regime de-risk OFF (fully invested below the 200dMA).
bdates, gross_on, net_on, turn, spxf, ndxf, _holds = E._edge_full(
    42, 20, 2e9, 0.50, 126, 0.25, 10.0, SIG)
_, gross_off, net_off, _, _, _, _ = E._edge_full(
    42, 20, 2e9, 0.50, 126, 1.0, 10.0, SIG)

pan = E.load_edge_panel(hold=42)
ma_on = pan.ma200_on
PPY = true_ppy(bdates)

# Plain momentum benchmark: same selection clock but the raw academic 3m-momentum
# top-N, equal-weight, no overlays / costs (inherits the canonical crash risk).
mom_holds = [E.select_topN(df, n=20, weights={"ret_63": 1.0}) for df in pan.panels]
mom = E.period_returns(pan, mom_holds)

years = np.array([d.year for d in bdates])

def block(name, r, bench, ppy):
    r, bench = np.asarray(r, float), np.asarray(bench, float)
    ex = r - bench
    return dict(name=name, n=len(r), mean=np.nanmean(r), ex=np.nanmean(ex),
                hit_abs=np.mean(r > 0), hit_vs=np.mean(r > bench),
                sh=ann_sharpe(r, ppy))

print("=" * 78)
print(f"Edge per-period analysis | hold=42d | {len(net_on)} periods | "
      f"~{PPY:.1f} periods/yr | {bdates[0].date()} -> {bdates[-1].date()}")
print(f"Market above 200dMA: {ma_on.mean()*100:.0f}% of periods")
print("=" * 78)

# ---- (1) CONDITIONAL PERFORMANCE (uses full-spec NET returns) --------------
print("\n(1) CONDITIONAL PERFORMANCE  -- Edge NET (full spec, regime ON)")
print("    bucket            n    Edge    S&P   excess  hitAbs  hit>SPX   Sharpe")
def crow(label, mask):
    if mask.sum() == 0:
        print(f"    {label:<16}{0:>4}      --"); return
    r, b = net_on[mask], spxf[mask]
    ex = r - b
    print(f"    {label:<16}{mask.sum():>4}{np.nanmean(r)*100:7.2f}%{np.nanmean(b)*100:6.2f}%"
          f"{np.nanmean(ex)*100:+7.2f}%{np.mean(r>0)*100:6.0f}%{np.mean(r>b)*100:8.0f}%"
          f"{ann_sharpe(r,PPY):8.2f}")
mkt_up = spxf >= 0
mkt_dn = spxf < 0
crow("market UP",      mkt_up)
crow("market DOWN",    mkt_dn)
crow("above 200dMA",   ma_on)
crow("below 200dMA",   ~ma_on)
crow("ALL",            np.ones(len(net_on), bool))

# Also show what the Edge does in down markets WITHOUT the regime cash crutch,
# to separate signal alpha from the regime overlay in the down bucket.
print("\n    [down-market detail] mean Edge return when S&P falls:")
print(f"      regime ON  (full spec) : {np.nanmean(net_on[mkt_dn])*100:+.2f}%  "
      f"(excess {np.nanmean((net_on-spxf)[mkt_dn])*100:+.2f}%)")
print(f"      regime OFF (signal only): {np.nanmean(net_off[mkt_dn])*100:+.2f}%  "
      f"(excess {np.nanmean((net_off-spxf)[mkt_dn])*100:+.2f}%)")

# ---- (2) SKEW / CRASH PROFILE ----------------------------------------------
print("\n(2) SKEW / CRASH PROFILE  (per-period returns)")
print("    series                 skew   worst    2nd     3rd    mean    std")
def srow(label, r):
    r = np.asarray(r, float); rs = np.sort(r[~np.isnan(r)])
    print(f"    {label:<20}{skew(r):7.2f}{rs[0]*100:7.1f}%{rs[1]*100:7.1f}%{rs[2]*100:7.1f}%"
          f"{np.nanmean(r)*100:6.2f}%{np.nanstd(r)*100:6.2f}%")
srow("Edge NET regimeON",  net_on)
srow("Edge NET regimeOFF", net_off)
srow("Edge GROSS regOFF",  gross_off)
srow("Plain momentum",     mom)
srow("S&P 500",            spxf)
srow("Nasdaq",             ndxf)

# ---- (3) CRISIS YEARS ------------------------------------------------------
def year_ret(r, y):
    m = years == y
    if m.sum() == 0:
        return None
    return float(np.cumprod(1 + np.nan_to_num(np.asarray(r, float)[m]))[-1] - 1)

print("\n(3) CRISIS-YEAR CALENDAR RETURNS  (compounded within-year periods)")
print("    year   EdgeON  EdgeOFF  PlainMom    S&P   Nasdaq   #per")
for y in (2008, 2009, 2020, 2022):
    n = int((years == y).sum())
    def f(x):
        v = year_ret(x, y); return "   --" if v is None else f"{v*100:+6.1f}%"
    print(f"    {y}  {f(net_on)} {f(net_off)}  {f(mom)} {f(spxf)} {f(ndxf)}    {n}")

# ---- (4) REGIME DECOMPOSITION ----------------------------------------------
print("\n(4) REGIME OVERLAY DECOMPOSITION  (full history, NET)")
print("    variant                CAGR   Sharpe   maxDD")
def drow(label, r):
    print(f"    {label:<22}{cagr(r,PPY)*100:6.1f}%{ann_sharpe(r,PPY):8.2f}{maxdd(r)*100:7.0f}%")
drow("regime ON  (0.25)",  net_on)
drow("regime OFF (1.0)",   net_off)
drow("S&P 500",            spxf)
print(f"\n    Periods where regime cash kicks in (below 200dMA): {(~ma_on).sum()} of {len(ma_on)}")
print(f"    In those periods: signal-only mean {np.nanmean(net_off[~ma_on])*100:+.2f}%  "
      f"vs regime-cash mean {np.nanmean(net_on[~ma_on])*100:+.2f}%  "
      f"(S&P {np.nanmean(spxf[~ma_on])*100:+.2f}%)")
