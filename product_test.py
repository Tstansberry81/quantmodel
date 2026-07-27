"""The PRODUCT, under its actual constraints. Basket <= 10, regime gate on.

Non-negotiables (fixed, not swept):
  * basket size <= 10 -- the product outputs a 10-name book with conviction
    weights and dollar allocations, so anything wider isn't shippable. Sizes
    below 10 are fair game.
  * the 200dMA regime gate stays on -- it earned its keep (turning it off blew
    the train drawdown from -22% to -62%).
  * long-only.

Everything else is what's being decided here: the market-cap floor, and
whether a free-cash-flow screen belongs on top of momentum.

WHY FCF IS BACK. In the single-factor battery FCF margin was the strongest
survivor -- the only factor whose OUT-of-sample t-stat (2.2) beat its
in-sample one -- and FCF yield was the best value proxy down-cap (+1.16%/reb
in micro, fading to +0.05% in large). Both got sidelined when momentum's
context result took over. They belong as a SCREEN rather than a ranker:
momentum decides what to buy, cash generation decides what to refuse.

Split: TRAIN 1999-2012 | TEST 2012-2020 | HOLDOUT 2021+.

Run:  .venv-mac/bin/python product_test.py [hold]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd

import edge_data as D
import edge_lib as E

COST_BPS, REGIME_EXPO = 10.0, 0.25
HOLDOUT, TEST = pd.Timestamp("2021-01-01"), pd.Timestamp("2012-09-01")

# screen -> (column, keep-the-top-half?) ; None = no screen
SCREENS = {
    "none":            None,
    "no-diluter":      ("share_issuance", False),   # keep LOW issuance
    "fcf margin":      ("fcf_margin", True),        # keep HIGH margin
    "fcf yield":       ("fcf_yield", True),
    "fcf marg+dilut":  ("fcf_margin+share_issuance", None),
}


def _apply_screen(d, name):
    """Halve the candidate pool on the screen before momentum ranks it."""
    if name == "none" or SCREENS.get(name) is None:
        return d
    if name == "fcf marg+dilut":
        m = pd.to_numeric(d.get("fcf_margin"), errors="coerce")
        i = pd.to_numeric(d.get("share_issuance"), errors="coerce")
        keep = ((m >= m.median()) & (i <= i.median())).fillna(False)
        return d[keep] if keep.sum() >= 20 else d
    col, high_good = SCREENS[name]
    v = pd.to_numeric(d.get(col), errors="coerce")
    keep = (v >= v.median()) if high_good else (v <= v.median())
    keep = keep.fillna(False)
    return d[keep] if keep.sum() >= 20 else d


def build(pan, n, floor, screen="none", signal="ret_12_1"):
    rets, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= floor]
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        d = d[np.isfinite(fwd)]
        if "fcf" in d.columns and "ev_d" in d.columns:      # ev is in millions
            ev = pd.to_numeric(d["ev_d"], errors="coerce") * 1e6
            d = d.assign(fcf_yield=pd.to_numeric(d["fcf"], errors="coerce") / ev.where(ev > 0))
        d = _apply_screen(d, screen)
        v = pd.to_numeric(d.get(signal), errors="coerce")
        ok = np.isfinite(v)
        d, v = d[ok], v[ok]
        if len(d) < n:
            rets.append(0.0); turn.append(0.0); continue
        pick = d.iloc[np.argsort(v.values)].tail(n)
        r = float(pd.to_numeric(pick["fwd_ret"], errors="coerce").mean())
        cur = set(pick["company_key"])
        turn.append(1 - len(cur & prev) / len(cur) if prev else 1.0)
        prev = cur
        if not pan.ma200_on[i]:                       # regime gate: always on
            r = REGIME_EXPO * r + (1 - REGIME_EXPO) * pan.rf_per
        rets.append(r)
    return np.array(rets) - (COST_BPS / 1e4) * np.array(turn)


def main(argv) -> int:
    hold = int(argv[1]) if len(argv) > 1 else 63
    pan = E.load_edge_panel(hold=hold, universe=20000)
    ppy = pan.ppy
    bd = pd.DatetimeIndex(pan.bdates)
    masks = {"TRAIN": bd < TEST, "TEST": (bd >= TEST) & (bd < HOLDOUT),
             "HOLDOUT": bd >= HOLDOUT}
    print(f"panel: {D.meta().get('source')} | {pan.T} rebalances | hold={hold}d | "
          f"basket <= 10, regime gate ON, long-only, net of {COST_BPS:.0f}bps")
    for k, m in masks.items():
        c, dd, sh = E.perf(pan.spxf[m], ppy)
        print(f"  S&P {k:<8} CAGR {c*100:5.1f}%  Sharpe {sh:4.2f}  maxDD {dd*100:5.0f}%")

    def row(label, net):
        cells = ""
        for m in masks.values():
            r = net[m]
            if len(r) < 3:
                cells += f"{'--':>22}"; continue
            c, dd, sh = E.perf(r, ppy)
            cb = E.perf(pan.spxf[m], ppy)[0]
            cells += f"{(c-cb)*100:>+9.1f}% Sh{sh:>5.2f}{dd*100:>7.0f}%"
        print(f"{label:<36}{cells}")

    hdr = f"\n{'configuration':<36}" + "".join(f"{k+' xs/Sh/DD':>22}" for k in masks)

    print(hdr); print("-" * len(hdr))
    for n in (5, 7, 8, 10):
        row(f"$10B, n={n}, no screen", build(pan, n, 1e10))

    print(hdr.replace("configuration", "SCREENS at n=10, $10B")); print("-" * len(hdr))
    for s in SCREENS:
        row(f"  {s}", build(pan, 10, 1e10, s))

    print(hdr.replace("configuration", "FLOOR x SCREEN at n=10")); print("-" * len(hdr))
    for fl, lbl in ((1e10, "$10B"), (2e9, "$2B"), (5e8, "$500M")):
        for s in ("no-diluter", "fcf margin", "fcf marg+dilut"):
            row(f"  {lbl:<6} {s}", build(pan, 10, fl, s))

    print(hdr.replace("configuration", "BEST-SHAPE at n=7 (tighter book)")); print("-" * len(hdr))
    for fl, lbl in ((1e10, "$10B"), (2e9, "$2B")):
        for s in ("no-diluter", "fcf marg+dilut"):
            row(f"  {lbl:<6} {s}", build(pan, 7, fl, s))

    print("\nCells: EXCESS CAGR vs the S&P, Sharpe, max drawdown. Screens halve "
          "the candidate pool BEFORE momentum ranks it, so momentum still picks "
          "the book -- cash generation only decides what it may not buy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
