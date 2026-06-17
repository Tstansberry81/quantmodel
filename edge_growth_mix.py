"""Edge growth-mix sweep: is there a middle ground between the rev-growth gate on vs off?

Instead of the binary gate (0 or all 20 names required to have >=15% YoY revenue
growth), build BLENDED baskets: K names drawn from the growth-gated pool (accel-
ranked, >=15% rev growth) + the remaining (20-K) from the pure-acceleration pool.
Sweep K = 0 (gate off) -> 20 (gate fully on) and see which mix maximizes CAGR /
alpha (excess vs S&P) and how risk (Sharpe, drawdown) moves.

Full Edge spec otherwise: hold=42 (~2mo), n=20, >=$2B liquidity floor, corr-cap 0.50,
200dMA regime (25% invested below the line), net of 10bps costs.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E

HOLD, N, FLOOR, CAP, GATE, REG, COST = 42, 20, 2e9, 0.50, 0.15, 0.25, 10.0
W = {"accel": 1.0}

print("Building panel ...")
pan = E.load_edge_panel(hold=HOLD)
ppy = pan.ppy
yr = pan.bdates.year


def blend_holds(k):
    """K growth-gated (>=15% rev) accel picks + fill to N from the pure-accel pool."""
    holds = []
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= FLOOR]
        asof = pan.bdates[i]
        gpick = []
        if k > 0:
            g = d[pd.to_numeric(d["rev_growth"], errors="coerce") >= GATE]
            if len(g) >= 1:
                gpick = E.corr_cap_select(g, asof=asof, n=k, weights=W, cap=CAP)
        fill = E.corr_cap_select(d, asof=asof, n=N + len(gpick), weights=W, cap=CAP)
        sel = list(gpick)
        for ck in fill:
            if len(sel) >= N:
                break
            if ck not in sel:
                sel.append(ck)
        holds.append(sel[:N])
    return holds


def net_returns(holds):
    gross = E.period_returns(pan, holds)
    r = np.where(pan.ma200_on, gross, REG * gross + (1 - REG) * pan.rf_per)
    turn, prev = [], None
    for h in holds:
        cur = set(h)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
    return r - (COST / 1e4) * np.array(turn)


def avg_growth_in_book(holds):
    """Average # of the 20 holdings that actually have >=15% YoY rev growth."""
    cnts = []
    for i, h in enumerate(holds):
        df = pan.panels[i]
        sub = df[df["company_key"].isin(h)]
        rg = pd.to_numeric(sub["rev_growth"], errors="coerce")
        cnts.append(int((rg >= GATE).sum()))
    return float(np.mean(cnts))


def win(r, k_periods):
    kk = min(k_periods, len(r))
    return E.perf(r[-kk:], ppy)              # (cagr, dd, sharpe)


sp_max = E.perf(pan.spxf, ppy)[0]
W1, W2, W5 = int(round(ppy)), int(round(2 * ppy)), int(round(5 * ppy))

print("\n" + "=" * 92)
print("GROWTH-MIX SWEEP  (K growth-gated names out of 20; alpha = CAGR - S&P CAGR)")
print("=" * 92)
print(f"{'mix':<16}{'MAX CAGR':>9}{'alpha':>7}{'Shrp':>6}{'maxDD':>7}"
      f"{'5Y':>7}{'2Y':>7}{'1Y':>8}{'  ~growth in book':>18}")
rows = []
for k in (0, 4, 8, 10, 12, 16, 20):
    holds = blend_holds(k)
    r = net_returns(holds)
    c, dd, sh = E.perf(r, ppy)
    c5 = win(r, W5)[0]; c2 = win(r, W2)[0]; c1 = win(r, W1)[0]
    gb = avg_growth_in_book(holds)
    rows.append((k, c, sh, dd, c5, c2, c1, gb))
    lbl = f"{k}/20 growth" + (" (off)" if k == 0 else " (full on)" if k == 20 else "")
    print(f"{lbl:<16}{c*100:8.1f}%{(c-sp_max)*100:+6.1f}%{sh:6.2f}{dd*100:6.0f}%"
          f"{c5*100:6.0f}%{c2*100:6.0f}%{c1*100:7.0f}%{gb:>14.0f}/20")

print(f"\nS&P 500 over MAX: {sp_max*100:.1f}% CAGR (the alpha baseline)")
# highlight the maximizers
best_max = max(rows, key=lambda x: x[1])
best_5y = max(rows, key=lambda x: x[4])
best_sh = max(rows, key=lambda x: x[2])
print(f"\nMax MAX-window CAGR/alpha : {best_max[0]}/20 growth  ({best_max[1]*100:.1f}% / +{(best_max[1]-sp_max)*100:.1f}%)")
print(f"Max 5Y CAGR              : {best_5y[0]}/20 growth  ({best_5y[4]*100:.1f}%)")
print(f"Max Sharpe (MAX)         : {best_sh[0]}/20 growth  ({best_sh[2]:.2f})")
print("\nNote: alpha here = excess CAGR vs the S&P (same benchmark each row, so it tracks CAGR).")
print("Sweeping the mix to pick a peak is mildly in-sample — prefer a mix that's good across")
print("ALL windows, not a knife-edge max. Survivorship/recent-momentum caveats still apply.")
