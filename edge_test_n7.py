"""The Edge -- stress-test the n=7 candidate before believing it.

edge_test_tail.py's grid says n=7 mix=75 @hold=42 beats the current n=10 product
in every window (MAX 25.9%/1.13/-25% vs 20.7%/1.03/-21%), consistent with the
independently-measured steep tail curve (top-5 >> top-10 >> top-20). But "best
cell of a grid" is the classic snooping trap, so before recommending anything:

  1. NEIGHBOR PLATEAU  n in {5,6,7,8,9,10}: real effects are plateaus, lucky
     cells are spikes.
  2. HALF-SPLIT        the n=7 net curve must work in BOTH halves (2005-2015 /
     2015-2026), compared against n=10's same split.
  3. COST SENSITIVITY  10/20/30 bps on the n=7 book.
  4. CORR-CAP SWEEP    cap in {0.40,0.50,0.60,None} at n=7 -- with fewer names
     the cap binds differently.

All full-spec net (floor + regime + growth-mix 75). No edge_lib/qmodel changes.
Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import edge_lib as E
import edge_test_skip as SK
from edge_test_skip import run_spec, window_row

W = {"accel": 1.0}


def halves(pan, net, label):
    ppy = pan.ppy; T = len(net); h = T // 2
    rows = []
    for name, seg in (("full", net), ("1st half", net[:h]), ("2nd half", net[h:])):
        c, dd, sh = E.perf(seg, ppy)
        sp = pan.spxf[:h] if name == "1st half" else (pan.spxf[h:] if name == "2nd half" else pan.spxf)
        cs, _, _ = E.perf(sp, ppy)
        rows.append(f"{name}: {c*100:5.1f}%/{sh:4.2f}/{dd*100:4.0f}% ({(c-cs)*100:+5.1f})")
    print(f"  {label:<18}" + "   ".join(rows))


def main():
    pan = E.load_edge_panel(hold=42)
    SK.GROWTH_MIX = 0.75

    print("=" * 108)
    print("1. NEIGHBOR PLATEAU -- n=5..10, mix=75, hold=42, net (plateau = real; spike = lucky)")
    print("=" * 108)
    nets = {}
    for n in (5, 6, 7, 8, 9, 10):
        SK.NBASKET = n
        net, avg_to = run_spec(pan, W)
        nets[n] = net
        print(window_row(pan, net, f"n={n}", avg_to))

    print("\n" + "=" * 108)
    print("2. HALF-SPLIT of the net curve (must work in both halves)")
    print("=" * 108)
    for n in (7, 10):
        halves(pan, nets[n], f"n={n} mix=75")

    print("\n" + "=" * 108)
    print("3. COST SENSITIVITY at n=7 (10/20/30 bps, one-way on turnover)")
    print("=" * 108)
    SK.NBASKET = 7
    for bps in (10.0, 20.0, 30.0):
        SK.COST_BPS = bps
        net, avg_to = run_spec(pan, W)
        print(window_row(pan, net, f"n=7 cost={int(bps)}bps", avg_to))
    SK.COST_BPS = 10.0

    print("\n" + "=" * 108)
    print("4. CORR-CAP SWEEP at n=7 (0.40 / 0.50 / 0.60 / off)")
    print("=" * 108)
    for cap in (0.40, 0.50, 0.60, None):
        SK.CORR_CAP = cap
        net, avg_to = run_spec(pan, W)
        print(window_row(pan, net, f"n=7 cap={cap}", avg_to))
    SK.CORR_CAP = 0.50

    print("\nDONE.")


if __name__ == "__main__":
    main()
