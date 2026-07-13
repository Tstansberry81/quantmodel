"""The Edge -- is the top-tail engine STABLE, and does it survive the real spec?

edge_test_attrib.py found the alpha engine: top-5/10 by accel in the $2B-floored
universe at hold=42 = +13.6%/yr t=3.18 (top-5 in-pool +13.0%/yr t=2.75), while
the growth gate adds only +1.4%/yr t=0.9 and accel's overall IC is ~0.

Two follow-ups before believing it:
  A. STABILITY -- the tail excess split first-half / second-half / ex-2020.
     A t=3.18 that lives in 2005-2015 is not tradeable.
  B. REAL-SPEC GRID -- full net product (floor + corr-cap + regime + 10bps):
     n in {5,7,10,15} x growth-mix in {0, 0.75} at hold=42, + n=10 rows at
     hold=21 for the clock question (Vision currently ships the 1M clock).
     Quantifies (i) how much tail alpha survives the corr cap, (ii) what the
     growth gate's risk-shape actually buys, (iii) concentration vs n.

Per-period cross-sectional stats in A (t-stats meaningful); B is the noisy
10-name book, read jointly with A. No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import t_stat
from edge_test_skip import run_spec, window_row   # full-spec net evaluator

GROWTH_THRESH, MCAP_FLOOR = 0.15, 2e9


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def tail_series(pan, k, gated):
    """Per-period excess of top-k-by-accel over its own pool mean."""
    out, dates = [], []
    for i, df in enumerate(pan.panels):
        d = df.dropna(subset=["fwd_ret"])
        d = d[_num(d, "pit_mcap") >= MCAP_FLOOR]
        if gated:
            d = d[_num(d, "rev_growth") >= GROWTH_THRESH]
        if len(d) < max(50, k * 3):
            continue
        s = _num(d, "accel")
        top = d.loc[s.nlargest(k).index]
        out.append(float(top["fwd_ret"].mean() - d["fwd_ret"].mean()))
        dates.append(pan.bdates[i])
    return np.array(out), pd.DatetimeIndex(dates)


def stability(pan, k, gated, label):
    x, dts = tail_series(pan, k, gated)
    ppy = pan.ppy
    half = len(x) // 2
    yrs = np.array([d.year for d in dts])
    segs = [("full", x), ("1st half", x[:half]), ("2nd half", x[half:]),
            ("ex-2020", x[yrs != 2020])]
    row = f"  {label:<34}"
    for name, seg in segs:
        row += f"{name}: {np.nanmean(seg)*ppy*100:+6.1f}%/yr t={t_stat(seg):5.2f}   "
    print(row)


def main():
    # ---------- A: stability of the tail engine ----------
    print("=" * 110)
    print("A. TAIL-ENGINE STABILITY at hold=42 (per-period excess over pool mean, annualized)")
    print("=" * 110)
    pan42 = E.load_edge_panel(hold=42)
    for k in (5, 10):
        stability(pan42, k, gated=False, label=f"top-{k} accel, floored, NO gate")
        stability(pan42, k, gated=True,  label=f"top-{k} accel, growth pool")

    # ---------- B: full-spec net grid ----------
    print("\n" + "=" * 110)
    print("B. FULL-SPEC NET GRID at hold=42 (floor+corrcap+regime+10bps) -- n x growth-mix")
    print("=" * 110)
    print(f"{'spec':<28}  window: CAGR/Sharpe/maxDD (excess vs S&P)")
    import edge_test_skip as SK
    for mix in (0.0, 0.75):
        for n in (5, 7, 10, 15):
            SK.GROWTH_MIX = mix; SK.NBASKET = n
            net, avg_to = run_spec(pan42, {"accel": 1.0})
            print(window_row(pan42, net, f"n={n:<3} mix={int(mix*100)}%", avg_to))
        print("-" * 110)

    print("\nC. CLOCK CHECK -- same product spec at hold=21 (what Vision ships)")
    pan21 = E.load_edge_panel(hold=21)
    for mix in (0.0, 0.75):
        SK.GROWTH_MIX = mix; SK.NBASKET = 10
        net, avg_to = run_spec(pan21, {"accel": 1.0})
        print(window_row(pan21, net, f"hold=21 n=10 mix={int(mix*100)}%", avg_to))

    print("\nDONE.")


if __name__ == "__main__":
    main()
