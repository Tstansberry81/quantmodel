"""Portfolio-level test of the changes the signal scan implies.

signal_scan.py (rank-IC + top-N tail on the survivorship-free Sharadar panel)
said three things, each measured on a fixed column with no fitting:

  * accel -- the shipped signal -- has a tail of +0.27%/rebalance at t=0.4 and
    is positive in only 45% of rebalances. It does not rank the cross-section.
  * The momentum family does: ret_12_1 (+1.88%, t=2.0) and ret_126 (+1.99%,
    t=2.2), both positive in BOTH halves of the sample.
  * rev_growth has a NEGATIVE tail (-1.61%, t=-2.5) in both halves -- yet the
    product forces 75% of the basket out of the high-revenue-growth pool.

So this runs the small number of configurations those findings imply. It is a
CONFIRMATION step, not a search: every config below is a hypothesis stated in
advance by the scan, and the count is deliberately tiny so the multiple-testing
burden stays interpretable.

Run:  .venv-mac/bin/python config_test.py
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")   # hold the universe rule fixed

import edge_data as D
import edge_lib as E

WINDOWS = ("5Y", "10Y", "20Y", "MAX")

CONFIGS = [
    ("A shipped: accel / mix .75 / n10", {"signal": {"accel": 1.0}, "growth_mix": 0.75, "n": 10}),
    ("B accel / mix 0 / n10",            {"signal": {"accel": 1.0}, "growth_mix": 0.0,  "n": 10}),
    ("C 12-1 mom / mix .75 / n10",       {"signal": {"ret_12_1": 1.0}, "growth_mix": 0.75, "n": 10}),
    ("D 12-1 mom / mix 0 / n10",         {"signal": {"ret_12_1": 1.0}, "growth_mix": 0.0,  "n": 10}),
    ("E 6m mom  / mix 0 / n10",          {"signal": {"ret_126": 1.0},  "growth_mix": 0.0,  "n": 10}),
    ("F 12-1 mom / mix 0 / n30",         {"signal": {"ret_12_1": 1.0}, "growth_mix": 0.0,  "n": 30}),
]


def main() -> int:
    print(f"panel: {D.meta().get('source')} | artifact={D.ARTIFACT_FILE} | "
          f"42d clock, net of 10bps, t+1 execution\n")
    hdr = f"{'config':<34}" + "".join(f"{w:>16}" for w in WINDOWS)
    print(hdr); print("-" * len(hdr))
    for label, spec in CONFIGS:
        spec = {"hold": 42, **spec}
        bt = E.run_edge_backtest(window="MAX", spec=spec)
        if not bt.get("ok"):
            print(f"{label:<34}  FAILED: {bt.get('reason')}"); continue
        by = {w["w"]: w for w in bt["windows"]}
        cells = ""
        for w in WINDOWS:
            if w in by:
                cells += f"{by[w]['excess']*100:>+9.1f}% Sh{by[w]['sharpe']:>5.2f}"
            else:
                cells += f"{'--':>16}"
        print(f"{label:<34}{cells}", flush=True)
    print("\nCells are EXCESS CAGR vs the S&P (model - benchmark) and Sharpe, per window.")
    print("Data: Sharadar SEP/SF1 point-in-time (ART, datekey), delisted INCLUDED; "
          "universe: top-1000 by PIT market cap, >= $2B; filing lag +1 trading day; "
          "rebalance 42 trading days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
