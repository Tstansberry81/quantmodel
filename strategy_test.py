"""The strategy the context scan implies — end to end, at varying breadth.

context_scan.py found momentum's edge is CONDITIONAL and the conditions agree
across four independent cuts:

    size    +3.15%/reb in $10B+ (t=2.6) vs -0.22% in micro. A large-cap effect.
    regime  +4.06% above the 200dMA (t=2.7), -1.64% below. An uptrend effect.
    sector  Technology +2.95% (t=2.6), Communication Services +2.75% (t=3.4);
            eight other sectors ~0. A dispersion effect.
    era     positive in 1999-2007, 2016-2020 and 2021+; dead only in 2008-2015.

So: long-only, large-cap, 12-1 momentum, invested only while the market is
above its 200-day average, optionally screening out serial diluters (share
issuance was the most consistent fundamental in the scan).

This tests it as an actual portfolio at breadth 10 -> 100, because the earlier
failure was never the signal, it was ten names carrying too much idiosyncratic
risk to convert a real edge into risk-adjusted return.

Split: TRAIN 1999-2012, TEST 2012-2020, HOLDOUT 2021+. The conditions above
were read off the whole sample, so TRAIN/TEST are not clean for them -- the
honest question is whether the shape survives into the holdout.

Run:  .venv-mac/bin/python strategy_test.py [hold]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd

import edge_data as D
import edge_lib as E

LARGE_FLOOR = 1e10          # the size bucket where momentum actually worked
COST_BPS = 10.0
REGIME_EXPO = 0.25          # exposure when the S&P is below its 200-day average
HOLDOUT = pd.Timestamp("2021-01-01")
TEST = pd.Timestamp("2012-09-01")


def build(pan, n, signal="ret_12_1", issuance_screen=False, regime=True):
    """Per-rebalance net returns for the long-only book."""
    rets, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= LARGE_FLOOR]
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        d = d[np.isfinite(fwd)]
        if len(d) < n * 2:
            rets.append(0.0); turn.append(0.0); continue
        if issuance_screen and "share_issuance" in d.columns:
            iss = pd.to_numeric(d["share_issuance"], errors="coerce")
            keep = (iss <= iss.median()).fillna(True)   # drop the diluting half
            if keep.sum() >= n * 2:
                d = d[keep]
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        v = pd.to_numeric(d[signal], errors="coerce")
        ok = np.isfinite(v)
        d, fwd, v = d[ok], fwd[ok], v[ok]
        if len(d) < n:
            rets.append(0.0); turn.append(0.0); continue
        pick = d.iloc[np.argsort(v.values)].tail(n)
        r = float(pd.to_numeric(pick["fwd_ret"], errors="coerce").mean())
        cur = set(pick["company_key"])
        turn.append(1 - len(cur & prev) / len(cur) if prev else 1.0)
        prev = cur
        if regime and not pan.ma200_on[i]:
            r = REGIME_EXPO * r + (1 - REGIME_EXPO) * pan.rf_per
        rets.append(r)
    net = np.array(rets) - (COST_BPS / 1e4) * np.array(turn)
    return net


def periods(pan):
    bd = pd.DatetimeIndex(pan.bdates)
    return {"TRAIN": bd < TEST, "TEST": (bd >= TEST) & (bd < HOLDOUT), "HOLDOUT": bd >= HOLDOUT}


def report(pan, label, net, masks, ppy):
    cells = ""
    for name, m in masks.items():
        r, b = net[m], pan.spxf[m]
        if len(r) < 3:
            cells += f"{'--':>22}"; continue
        c, dd, sh = E.perf(r, ppy)
        cb = E.perf(b, ppy)[0]
        cells += f"{(c-cb)*100:>+9.1f}% Sh{sh:>5.2f}{dd*100:>7.0f}%"
    print(f"{label:<34}{cells}")


def main(argv) -> int:
    hold = int(argv[1]) if len(argv) > 1 else 63
    pan = E.load_edge_panel(hold=hold, universe=5000)
    ppy, masks = pan.ppy, periods(pan)
    print(f"panel: {D.meta().get('source')} | {pan.T} rebalances "
          f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()} | hold={hold}d")
    print(f"universe: >= ${LARGE_FLOOR/1e9:.0f}B market cap, long-only, "
          f"net of {COST_BPS:.0f}bps\n")
    for name, m in masks.items():
        c, dd, sh = E.perf(pan.spxf[m], ppy)
        print(f"  S&P {name:<8} CAGR {c*100:5.1f}%  Sharpe {sh:4.2f}  maxDD {dd*100:5.0f}%")
    hdr = (f"\n{'configuration':<34}" +
           "".join(f"{k+' xs/Sh/DD':>22}" for k in masks))
    print(hdr); print("-" * len(hdr))
    for n in (10, 20, 30, 50, 100):
        report(pan, f"12-1 mom, n={n}, regime on", build(pan, n), masks, ppy)
    print()
    for n in (20, 50):
        report(pan, f"  + no-diluter screen, n={n}",
               build(pan, n, issuance_screen=True), masks, ppy)
    for n in (20, 50):
        report(pan, f"  regime OFF (always in), n={n}",
               build(pan, n, regime=False), masks, ppy)
    print("\nCells are EXCESS CAGR vs the S&P, Sharpe, and max drawdown.")
    print("Data: Sharadar point-in-time, delisted INCLUDED; 12-1 momentum; "
          f"exposure cut to {REGIME_EXPO:.0%} when the S&P is below its 200dMA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
