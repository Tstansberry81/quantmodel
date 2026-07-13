"""The Edge -- do the OTHER dials shift at n=7?

The n=7 candidate passed all gates (plateau, halves, costs, corr-cap,
survivorship). But growth-mix=75 and hold=42 were tuned on 10-20-name books;
re-check the remaining dials at the new basket size:

  1. GROWTH-MIX sweep {0,25,50,75,100} at n=7, hold=42
  2. HOLD sweep {21,42,63} at n=7, mix=75 (each with its own ppy)
  3. REGIME exposure {0.0, 0.25, 0.50, off} at n=7, hold=42, mix=75

Full spec net (floor + corr-cap 0.5 + 10bps). No edge_lib/qmodel changes.
Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import edge_lib as E
import edge_test_skip as SK
from edge_test_skip import run_spec, window_row

W = {"accel": 1.0}


def main():
    SK.NBASKET = 7

    print("=" * 108)
    print("1. GROWTH-MIX sweep at n=7, hold=42")
    print("=" * 108)
    pan = E.load_edge_panel(hold=42)
    for mix in (0.0, 0.25, 0.50, 0.75, 1.0):
        SK.GROWTH_MIX = mix
        net, avg_to = run_spec(pan, W)
        print(window_row(pan, net, f"mix={int(mix*100)}%", avg_to))
    SK.GROWTH_MIX = 0.75

    print("\n" + "=" * 108)
    print("2. HOLD sweep at n=7, mix=75 (each clock annualized with its own ppy)")
    print("=" * 108)
    for hold in (21, 42, 63):
        p = E.load_edge_panel(hold=hold)
        net, avg_to = run_spec(p, W)
        print(window_row(p, net, f"hold={hold} (~{round(hold/21)}M)", avg_to))

    print("\n" + "=" * 108)
    print("3. REGIME exposure below the 200dMA at n=7, hold=42, mix=75")
    print("=" * 108)
    for expo in (0.0, 0.25, 0.50, None):
        SK.REGIME_EXPO = 1.0 if expo is None else expo   # None = overlay off (100% invested)
        net, avg_to = run_spec(pan, W)
        label = "off (100%)" if expo is None else f"expo={int(expo*100)}%"
        print(window_row(pan, net, label, avg_to))
    SK.REGIME_EXPO = 0.25

    print("\nDONE.")


if __name__ == "__main__":
    main()
