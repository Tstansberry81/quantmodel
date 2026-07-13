"""Rigor check on the CONTINUOUS-REGIME upgrade (evaluate 200dMA daily, not per-rebalance).

The round-3 timing test found that applying the EXISTING 200dMA regime daily
(instead of freezing it per 42d rebalance) cuts drawdown ~4-8pts at tied Sharpe.
Because it is the SAME rule/data applied more often (not a data-mined new signal),
the multiple-testing risk is low; the real questions are:
  * does the excess vs S&P stay significant (Newey-West t on daily excess)?
  * does the Sharpe survive deflation for the trials run (DSR)?
  * is the drawdown improvement ROBUST across both halves + not just one crisis?
The half-split + crisis-window drawdowns are the decisive checks: a DD win that
lives entirely in 2008 is fragile; one that shows up in both decades is real.

Run: .venv/Scripts/python.exe edge_test_timing_rigor.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import edge_lib as E
from edge_rigor import newey_west_t, deflated_sharpe
from edge_test_timing import book_daily, market_exposures, apply_rule, metrics, RF_D

# annual Sharpes observed across the round-3 timing sweep -> trial dispersion
TRIAL_SR_ANNUAL = [0.94, 0.85, 0.94, 0.91, 0.87, 0.76, 0.92, 0.98, 0.87,
                   0.90, 0.81, 0.91, 0.88, 0.84, 0.75, 0.90, 0.94, 0.87]
N_TRIALS = 150 + 40                 # prior campaign + round-3 timing/lowvol trials
CRISES = {"GFC 08-09": ("2008-06-01", "2009-06-30"),
          "COVID 2020": ("2020-02-01", "2020-05-31"),
          "2022 bear": ("2022-01-01", "2022-12-31")}


def maxdd(r):
    eq = np.cumprod(1 + np.nan_to_num(r)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def rigor_row(name, daily, spx):
    excess = daily.values - spx.values
    t_nw, T, L = newey_west_t(excess)
    var_sr = float(np.var(np.array(TRIAL_SR_ANNUAL) / np.sqrt(252)))
    dsr, sr, sr_star, g3, g4, _ = deflated_sharpe(daily.values, N_TRIALS, var_sr)
    print(f"  {name:<16} NW t(excess/day)={t_nw:+.2f} ({'sig' if abs(t_nw)>2 else 'ns'})"
          f"  ann-SR {sr*np.sqrt(252):.2f}  DSR {dsr:.3f} ({'PASS' if dsr>0.95 else 'FAIL'})")


def main():
    for n in (10, 7):
        print("\n" + "=" * 78)
        print(f"CONTINUOUS-REGIME RIGOR -- n={n}, hold={E.EDGE_SPEC['hold']}, daily/staggered")
        print("=" * 78)
        prod, spx = book_daily(n, E.EDGE_SPEC["regime_expo"])     # per-rebalance 200dMA@0.25
        pure, _ = book_daily(n, 1.0)                              # pure book (overlay canvas)
        exps = market_exposures(pure.index)
        prods = prod.reindex(pure.index).fillna(0.0)
        spx_al = spx.reindex(pure.index).fillna(0.0)

        def overlay(key):
            e = exps[key].reindex(pure.index).ffill().fillna(1.0)
            return pd.Series(e.values * pure.values + (1 - e.values) * RF_D
                             - 1e-4 * e.diff().abs().fillna(0).values, index=pure.index)
        rules = {"CONTINUOUS": overlay("d200_25"), "PANIC": overlay("panic")}

        for lab, series in [("PRODUCT", prods)] + list(rules.items()):
            c, sh, so, dd = metrics(series.values)
            print(f"  {lab:<16} CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  Sortino {so:.2f}  maxDD {dd*100:.0f}%")
        print("  --- significance / deflation ---")
        rigor_row("PRODUCT", prods, spx_al)
        for lab, series in rules.items():
            rigor_row(lab, series, spx_al)

        print("  --- half-split (is the DD win in BOTH halves?) ---")
        h = len(pure) // 2
        for lab, series in [("PRODUCT", prods)] + list(rules.items()):
            r = series.values
            _, s1, _, d1 = metrics(r[:h]); _, s2, _, d2 = metrics(r[h:])
            print(f"  {lab:<16} 1H Sharpe {s1:.2f} maxDD {d1*100:4.0f}%   2H Sharpe {s2:.2f} maxDD {d2*100:4.0f}%")

        print("  --- crisis-window drawdowns (concentration check) ---")
        for cname, (a, b) in CRISES.items():
            m = (pure.index >= a) & (pure.index <= b)
            dprod = maxdd(prods.values[m])
            deltas = "   ".join(f"{lab} {maxdd(s.values[m])*100:4.0f}% ({(maxdd(s.values[m])-dprod)*100:+.0f})"
                                for lab, s in rules.items())
            print(f"  {cname:<16} PRODUCT {dprod*100:5.0f}%   {deltas}")


if __name__ == "__main__":
    main()
