"""Which rebalance clock: 1-month, 2-month or 6-month?

Three candidates, one structural choice, decided on ALL-PERIOD consistency
rather than the single best number. That distinction is what keeps this from
being overfitting: the clock has to be set somehow, three is the whole
sensible space, and there is a prior -- 12-1 momentum is a slow signal, so a
fast clock re-trades the same information and pays costs for it.

Config is the recommended product shape and is held fixed:
    $10B floor | n=10 | 12-1 momentum | regime gate ON | FCF-margin screen

Turnover and the cost drag are printed, because that is the mechanism by
which a fast clock loses even when its gross signal is identical.

Run:  .venv-mac/bin/python clock_test.py
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd

import edge_data as D
import edge_lib as E

FLOOR, N = 1e10, 10
COST_BPS, REGIME_EXPO = 10.0, 0.25
HOLDOUT, TEST = pd.Timestamp("2021-01-01"), pd.Timestamp("2012-09-01")
CLOCKS = ((21, "1 month"), (42, "2 month"), (126, "6 month"))


def run(hold, screen=True):
    pan = E.load_edge_panel(hold=hold, universe=20000)
    rets, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= FLOOR]
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        d = d[np.isfinite(fwd)]
        if screen and "fcf_margin" in d.columns:
            m = pd.to_numeric(d["fcf_margin"], errors="coerce")
            keep = (m >= m.median()).fillna(False)
            if keep.sum() >= 20:
                d = d[keep]
        v = pd.to_numeric(d.get("ret_12_1"), errors="coerce")
        ok = np.isfinite(v)
        d, v = d[ok], v[ok]
        if len(d) < N:
            rets.append(0.0); turn.append(0.0); continue
        pick = d.iloc[np.argsort(v.values)].tail(N)
        r = float(pd.to_numeric(pick["fwd_ret"], errors="coerce").mean())
        cur = set(pick["company_key"])
        turn.append(1 - len(cur & prev) / len(cur) if prev else 1.0)
        prev = cur
        if not pan.ma200_on[i]:
            r = REGIME_EXPO * r + (1 - REGIME_EXPO) * pan.rf_per
        rets.append(r)
    turn = np.array(turn)
    net = np.array(rets) - (COST_BPS / 1e4) * turn
    return pan, net, turn


def main() -> int:
    print(f"config: >= ${FLOOR/1e9:.0f}B | n={N} | 12-1 momentum | regime ON | "
          f"FCF-margin screen | net of {COST_BPS:.0f}bps\n")
    hdr = (f"{'clock':<12}{'rebals':>7}{'turn/reb':>10}{'cost/yr':>9}"
           f"{'TRAIN xs':>10}{'TEST xs':>9}{'HOLD xs':>9}{'HOLD Sh':>9}{'HOLD DD':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for hold, label in CLOCKS:
        pan, net, turn = run(hold)
        ppy = pan.ppy
        bd = pd.DatetimeIndex(pan.bdates)
        masks = {"TRAIN": bd < TEST, "TEST": (bd >= TEST) & (bd < HOLDOUT),
                 "HOLDOUT": bd >= HOLDOUT}
        xs = {}
        for k, m in masks.items():
            if m.sum() < 3:
                xs[k] = (0, 0, 0); continue
            c, dd, sh = E.perf(net[m], ppy)
            cb = E.perf(pan.spxf[m], ppy)[0]
            xs[k] = (c - cb, sh, dd)
        cost_yr = turn.mean() * (COST_BPS / 1e4) * ppy
        print(f"{label:<12}{pan.T:>7}{turn.mean():>9.0%}{cost_yr:>8.2%}"
              f"{xs['TRAIN'][0]*100:>+9.1f}%{xs['TEST'][0]*100:>+8.1f}%"
              f"{xs['HOLDOUT'][0]*100:>+8.1f}%{xs['HOLDOUT'][1]:>9.2f}"
              f"{xs['HOLDOUT'][2]*100:>8.0f}%")
        rows.append((label, xs))
    print("\nChosen on consistency: the clock that is positive in the MOST "
          "periods, with the cost drag it actually pays. A clock that wins one "
          "period and loses two has been picked by the period, not the logic.")
    for label, xs in rows:
        pos = sum(1 for k in ("TRAIN", "TEST", "HOLDOUT") if xs[k][0] > 0)
        print(f"  {label:<10} positive in {pos}/3 periods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
