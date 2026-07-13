"""The Edge -- low-volatility / betting-against-beta CROSS-SECTIONAL test (round 3).

The one shortlist phenomenon that is a cross-sectional signal, not a timing
overlay. Tests whether tilting the accel selection toward LOW beta / LOW realized
vol improves the book. Both inputs (beta, vol_21) are ALREADY in the panel -- no
new data. Runs through the real staggered/daily product path via _edge_daily with
a tilted signal_key, so only the selection tilt differs from EDGE_SPEC.

Prior expectation (memory edge-alpha-literature): low-vol/ivol was weak-to-negative
on this universe in the price campaign -> expect a re-reject. This confirms cleanly.

Usage: .venv/Scripts/python.exe edge_test_lowvol.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E

S = E.EDGE_SPEC


def run(n, signal_key):
    idx, model, spx, ndx, turn, holds = E._edge_daily(
        S["hold"], n, S["mcap_floor"], S["corr_cap"], S["corr_lookback"],
        S["regime_expo"], S["cost_bps"], signal_key,
        S["growth_mix"], S["growth_thresh"], S["stagger"])
    r = np.asarray(model, float)
    c, dd, sh = E._perf_daily(r)
    return c, sh, E.sortino(r, ppy=252.0), dd


def main():
    variants = [
        ("baseline (accel)", (("accel", 1.0),)),
        ("accel - 0.25*vol", (("accel", 1.0), ("vol_21", -0.25))),
        ("accel - 0.50*vol", (("accel", 1.0), ("vol_21", -0.50))),
        ("accel - 1.0*vol",  (("accel", 1.0), ("vol_21", -1.00))),
        ("accel - 0.25*beta", (("accel", 1.0), ("beta", -0.25))),
        ("accel - 0.50*beta", (("accel", 1.0), ("beta", -0.50))),
        ("accel - 1.0*beta",  (("accel", 1.0), ("beta", -1.00))),
    ]
    for n in (7, 10):
        print("\n" + "=" * 74)
        print(f"LOW-VOL / BAB cross-sectional tilt -- n={n}, hold={S['hold']}, staggered/daily")
        print("=" * 74)
        print(f"{'variant':<20}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}")
        print("-" * 74)
        base = None
        for lab, sk in variants:
            c, sh, so, dd = run(n, sk)
            if base is None:
                base = (c, sh, so, dd); tag = ""
            else:
                tag = "  <-- better" if (sh > base[1] + 1e-9 and dd >= base[3] - 1e-9) else ""
            print(f"{lab:<20}{c*100:>7.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%{tag}")
        print("-" * 74)


if __name__ == "__main__":
    main()
