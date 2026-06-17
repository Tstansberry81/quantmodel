"""Is the growth-mixed Edge's edge REAL? Survivorship + stability + factor tests.

Runs the validation battery on the new default (growth_mix = 0.75) Edge:
  1. SURVIVORSHIP — full pool vs active-only ("today's survivors"); the gap is a
     measurable LOWER BOUND on survivorship inflation.
  2. STABILITY — first half vs second half vs ex-2020; is the edge consistent or a
     lucky window?
  3. FACTOR ATTRIBUTION — regress the model's net excess return on in-universe
     market + size + momentum + GROWTH factors (Newey-West). If alpha survives once
     you control for the momentum AND growth premia, it's real selection; if it
     collapses, the model is just harvesting buyable factors.

Full Edge spec: hold=42, n=20, >=$2B floor, corr-cap 0.50, 200dMA regime 25%, 10bps.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import tech_bias_lib as TB                 # ols_nw (Newey-West)
from qmodel import engine

HOLD, N, FLOOR, CAP, THRESH, REG, COST = 42, 20, 2e9, 0.50, 0.15, 0.25, 10.0
W = {"accel": 1.0}

print("Building panel ...")
pan = E.load_edge_panel(hold=HOLD)
ppy = pan.ppy
yr = pan.bdates.year
data = engine._load_bt_data()
INACTIVE = {ck for ck, b in data.items()
            if b.get("meta", {}).get("trading_status", "Active") != "Active"}


def holds(mix, active_only=False):
    out = []
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= FLOOR]
        if active_only:
            d = d[~d["company_key"].isin(INACTIVE)]
        out.append(E._blend_select(d, pan.bdates[i], N, W, CAP, 126, mix, THRESH))
    return out


def net(hlist):
    gross = E.period_returns(pan, hlist)
    r = np.where(pan.ma200_on, gross, REG * gross + (1 - REG) * pan.rf_per)
    turn, prev = [], None
    for h in hlist:
        cur = set(h)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
    return r - (COST / 1e4) * np.array(turn)


def cagr(r):
    eq = np.cumprod(1 + np.nan_to_num(r)); return eq[-1] ** (ppy / len(r)) - 1 if len(r) and eq[-1] > 0 else -1


W1, W2, W5 = int(round(ppy)), int(round(2 * ppy)), int(round(5 * ppy))
base = net(holds(0.75))
sp = pan.spxf

# ============================ 1. SURVIVORSHIP ================================
print("\n" + "=" * 74)
print("1. SURVIVORSHIP — full pool vs active-only (mix=75%); gap = lower bound")
print("=" * 74)
act = net(holds(0.75, active_only=True))
print(f"  inactive names in pool: {len(INACTIVE)} of {len(data)}")
print(f"  {'window':<6}{'full pool':>11}{'active-only':>13}{'gap':>9}")
for wn, k in (("1Y", W1), ("2Y", W2), ("5Y", W5), ("MAX", len(base))):
    cf, ca = cagr(base[-k:]), cagr(act[-k:])
    print(f"  {wn:<6}{cf*100:10.1f}%{ca*100:12.1f}%{(cf-ca)*100:+8.1f}%")
print("  active-only ~ 'today's survivors' (most biased); the truly-missing dead")
print("  names would drag the honest number BELOW the full-pool line.")

# ============================ 2. STABILITY ==================================
print("\n" + "=" * 74)
print("2. STABILITY — is the edge a lucky window? (mix=75%, alpha = excess vs S&P)")
print("=" * 74)
T = len(base); half = T // 2
def seg(mask, label):
    c = cagr(base[mask]); cs = cagr(sp[mask])
    print(f"  {label:<26} CAGR {c*100:6.1f}%   S&P {cs*100:5.1f}%   alpha {(c-cs)*100:+6.1f}%")
seg(np.arange(T) < half, f"first half {pan.bdates[0].date()}..{pan.bdates[half-1].date()}")
seg(np.arange(T) >= half, f"second half {pan.bdates[half].date()}..{pan.bdates[-1].date()}")
seg(yr != 2020, "excluding 2020")
# per-year hit-rate vs S&P
wins = sum(1 for y in set(yr) if np.prod(1+np.nan_to_num(base[yr==y]))-1 > np.prod(1+np.nan_to_num(sp[yr==y]))-1)
print(f"  beats S&P in {wins}/{len(set(yr))} calendar years")

# ====================== 3. FACTOR ATTRIBUTION ===============================
print("\n" + "=" * 74)
print("3. FACTOR ATTRIBUTION — real alpha, or just momentum+growth premia? (mix=75%)")
print("=" * 74)
def ls(df, col, higher_long=True):
    g = df.dropna(subset=[col, "fwd_ret"])
    if len(g) < 9:
        return 0.0
    v = pd.to_numeric(g[col], errors="coerce"); f = g["fwd_ret"].to_numpy(float)
    q1, q2 = v.quantile(1/3), v.quantile(2/3)
    hi, lo = f[(v >= q2).to_numpy()], f[(v <= q1).to_numpy()]
    if len(hi) == 0 or len(lo) == 0:
        return 0.0
    return float(np.nanmean(hi) - np.nanmean(lo)) if higher_long else float(np.nanmean(lo) - np.nanmean(hi))

MKT, SMB, MOM, GRW = [], [], [], []
for i, df in enumerate(pan.panels):
    d = df[df["pit_mcap"] >= FLOOR]
    MKT.append(sp[i] - pan.rf_per)
    SMB.append(ls(d, "pit_mcap", higher_long=False))   # small minus big
    MOM.append(ls(d, "accel", higher_long=True))       # acceleration premium
    GRW.append(ls(d, "rev_growth", higher_long=True))  # growth premium
F = {k: np.array(v) for k, v in [("MKT", MKT), ("SMB", SMB), ("MOM", MOM), ("GRW", GRW)]}
y = base - pan.rf_per

def attr(label, cols):
    X = np.column_stack([np.ones(T)] + [F[c] for c in cols])
    beta, se, t = TB.ols_nw(y, X, L=6)
    a_ann = (1 + beta[0]) ** ppy - 1
    tag = "** survives" if abs(t[0]) > 2 else "NOT significant" if abs(t[0]) < 1.65 else "~ marginal"
    print(f"  [{label}]  alpha {a_ann*100:+.1f}%/yr  t={t[0]:+.2f}  {tag}")
    for j, c in enumerate(cols, 1):
        print(f"        {c:<5} loading {beta[j]:+.2f}  t={t[j]:+.2f}")

attr("vs MKT", ["MKT"])
attr("vs MKT+SMB+MOM", ["MKT", "SMB", "MOM"])
attr("vs MKT+SMB+MOM+GROWTH", ["MKT", "SMB", "MOM", "GRW"])
print("\nNote: in-universe factors built from the same pool can over-absorb the edge")
print("(stricter than published factors). Numbers are gross of the survivorship haircut.")
