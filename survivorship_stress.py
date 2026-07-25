"""Survivorship stress suite for the Edge (product config: 42d / 10 / 75% mix).

The backtest's survivorship handicap has three distinct components:

  1. MISSING DEAD NAMES -- fiscal.ai carries ~111 inactive names; the hundreds
     of index members that died pre-~2015 have no data at all. Not fixable in
     code; bounded here by comparing universes.
  2. UNIVERSE DEFINITION -- the top-1000-by-PIT-mcap proxy contains only names
     the cache knows, i.e. mostly survivors. data/sp500_membership.csv
     (S&P 500 membership history, 1996->present; fetch_membership.py) gives
     TRUE point-in-time membership; the priced subset of it is an
     honestly-constructed (if large-cap-only) universe.
  3. THE DROP RULE -- load_edge_panel excludes any name whose price series ends
     inside the forward hold window, so the model never holds a stock through
     its delisting. The EDGE_DELIST_HAIRCUT knob includes those names with
     returns truncated at the last trade plus a haircut.

This script runs the product config across five variants:

  A. proxy top-1000                  (the product baseline)
  B. proxy top-500                   (size-class control for C)
  C. true-PIT S&P 500 membership     (priced subset; see coverage note)
  D. proxy top-1000, delist truncate (haircut 0%)
  E. proxy top-1000, delist -30%     (stress bound)

Read A vs B as the size/breadth effect, B vs C as survivorship + index-committee
effects at matched size class, and A vs D/E as the drop-rule flattery bound
given the delisted data we actually have.

COVERAGE CAVEAT: the priced share of true members is ~41% in 1998, ~60% in
2010, ~94% in 2026 (missing names are dead pre-2015, plus foreign-domiciled
members the US-only universe build excludes). Variant C therefore still holds
survivors disproportionately in early years -- it reduces, not eliminates.

Run:  .venv-mac/bin/python survivorship_stress.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PIT_MEMBERSHIP_CSV",
                      str(Path(__file__).resolve().parent / "data" / "sp500_membership.csv"))

import numpy as np

import edge_lib as E

HOLD = 42
N = 10
MCAP_FLOOR = 2e9
CORR_CAP = 0.50
CORR_LOOKBACK = 126
REGIME_EXPO = 0.25
COST_BPS = 10.0
SIGNAL = {"accel": 1.0}
GROWTH_MIX = 0.75
GROWTH_THRESH = 0.15
WINDOWS = ("10Y", "20Y")


def _walk(pan):
    """Replicate _edge_full's selection walk on an arbitrary panel."""
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= MCAP_FLOOR]
        cks = E._blend_select(d, pan.bdates[i], N, SIGNAL, CORR_CAP,
                              CORR_LOOKBACK, GROWTH_MIX, GROWTH_THRESH)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
        holds.append(cks)
    gross = E.period_returns(pan, holds)
    gross = np.where(pan.ma200_on, gross, REGIME_EXPO * gross + (1 - REGIME_EXPO) * pan.rf_per)
    net = gross - (COST_BPS / 1e4) * np.array(turn)
    return net


def run_variant(label: str, universe: int = 1000, use_pit: bool = False,
                haircut: float | None = None):
    E.USE_PIT_UNIVERSE = use_pit
    E.DELIST_HAIRCUT = haircut
    E.load_edge_panel.cache_clear()
    pan = E.load_edge_panel(hold=HOLD, universe=universe)
    net = _walk(pan)
    T = len(net)
    out = [label.ljust(34)]
    for w in WINDOWS:
        # rebalance count, not trading days: this harness measures on the
        # per-rebalance series, while E.window_k counts days for the daily curve
        k = T if w == "MAX" else min(int(round(int(w[:-1]) * pan.ppy)), T)
        sl = slice(T - k, T)
        cagr, dd, sh = E.perf(net[sl], pan.ppy)
        sp_cagr = E.perf(pan.spxf[sl], pan.ppy)[0]
        out.append(f"{w}: {cagr*100:5.1f}% (xs {(cagr-sp_cagr)*100:+5.1f}%) Sh {sh:.2f} DD {dd*100:5.1f}%")
    print("  ".join(out), flush=True)


if __name__ == "__main__":
    print(f"Edge survivorship stress -- product config {HOLD}d / {N} names / "
          f"{int(GROWTH_MIX*100)}% growth mix, net of {COST_BPS:.0f}bps\n")
    run_variant("A proxy top-1000 (baseline)")
    run_variant("B proxy top-500 (size control)", universe=500)
    run_variant("C true-PIT S&P500 (priced subset)", use_pit=True)
    run_variant("D proxy + delist truncate (0%)", haircut=0.0)
    run_variant("E proxy + delist haircut (-30%)", haircut=-0.30)
    print("\nA-B = size/breadth | B-C = survivorship+committee at matched size | "
          "A-D/E = drop-rule flattery bound")
