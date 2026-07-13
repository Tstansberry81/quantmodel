"""The Edge -- verdict check on the #4 interaction meta-model (edge_test_interactions).

The meta-labeling book (interaction-EN re-ranks within the accel top-K pool) beat
accel in the OOS backtest, but OOS IC was insignificant (t=0.95) -> likely tail
luck. Decisive checks before wire-or-kill:
  1. HALF-SPLIT: does the meta Sharpe edge over accel hold in BOTH decades?
  2. CRISIS: 2020 / 2022 behavior vs accel.
  3. POOL SENSITIVITY: is the win specific to top-40, or robust across K in {20,30,40,50}?

A real edge survives both halves AND isn't cutoff-specific. Single-decade or
single-K success = luck -> kill. Reuses the #4 harness. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import edge_lib as E
import edge_test_metamodel as MM
import edge_test_interactions as INT

FLOOR = 2e9


def perf(r, ppy):
    c, dd, sh = E.perf(np.asarray(r, float), ppy)
    return c, sh, dd


def main():
    pan = E.load_edge_panel(hold=42); ppy = pan.ppy
    periods = INT.prep_interactions(pan)
    preds, _ = MM.walk_forward(periods)
    oos = [i for i, p in enumerate(periods) if p is not None and preds[i] is not None]
    yrs = np.array([pan.bdates[i].year for i in oos])
    half = len(oos) // 2

    for n in (7, 10):
        # accel baseline over the same OOS periods
        acc = []
        for i in oos:
            dd = pan.panels[i][pan.panels[i]["pit_mcap"] >= FLOOR]
            acc.append(E.corr_cap_select(dd, asof=pan.bdates[i], n=n, weights={"accel": 1.0},
                                         cap=0.5, lookback=126))
        r_ac, _ = MM.net_from_holds(pan, acc, oos)
        r_ac = np.asarray(r_ac)
        ca, sa, da = perf(r_ac, ppy)
        c1a, s1a, _ = perf(r_ac[:half], ppy); c2a, s2a, _ = perf(r_ac[half:], ppy)
        print(f"\n{'='*92}\nn={n}  accel baseline: full {ca*100:.1f}%/{sa:.2f}/{da*100:.0f}%   "
              f"1st {c1a*100:.1f}%/{s1a:.2f}  2nd {c2a*100:.1f}%/{s2a:.2f}\n{'='*92}")
        print(f"{'meta pool K':<14}{'full CAGR/Sh/DD':>20}{'1st half Sh':>13}{'2nd half Sh':>13}"
              f"{'2020':>8}{'2022':>8}{'  verdict'}")
        for K in (20, 30, 40, 50):
            MM.ACCEL_POOL = K
            h, idx = MM.book_from_scores(periods, preds, n, meta=True)
            r, _ = MM.net_from_holds(pan, h, idx)
            r = np.asarray(r)
            c, s, d = perf(r, ppy)
            _, s1, _ = perf(r[:half], ppy); _, s2, _ = perf(r[half:], ppy)
            y20 = float(np.prod(1 + r[yrs == 2020]) - 1) if (yrs == 2020).any() else float("nan")
            y22 = float(np.prod(1 + r[yrs == 2022]) - 1) if (yrs == 2022).any() else float("nan")
            beats_both = (s1 > s1a) and (s2 > s2a)
            v = "SURVIVES (both halves)" if beats_both else ("one-half only" if (s1 > s1a) or (s2 > s2a) else "loses both")
            print(f"K={K:<12}{c*100:6.1f}%/{s:.2f}/{d*100:.0f}%{s1:>13.2f}{s2:>13.2f}"
                  f"{y20*100:>7.0f}%{y22*100:>7.0f}%   {v}")
        print(f"   (accel crisis: 2020 {float(np.prod(1+r_ac[yrs==2020])-1)*100:.0f}%  "
              f"2022 {float(np.prod(1+r_ac[yrs==2022])-1)*100:.0f}%)")
    print("\nDONE.  Graduates only if meta beats accel Sharpe in BOTH halves across most K.")


if __name__ == "__main__":
    main()
