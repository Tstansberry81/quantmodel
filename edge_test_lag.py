"""The Edge -- execution-lag VALIDITY test (GPT audit 2.3).

The current backtest forms the signal on close t AND enters at close t -- the same
bar. That is a look-ahead: you cannot compute the whole cross-section and also
trade the close it was computed from. A real medium-horizon signal must survive a
one-day lag. This tests the FULL-SPEC Edge (accel top-tail + $2B floor + corr-cap
+ 200dMA regime + 10bps) under three execution assumptions, holding selection
fixed (signal still formed on close t -- only the ENTRY changes):

  A. same-close (baseline)   enter at close t      -> exit close t+H
  B. close t+1 (lagged)      enter at close t+1    -> exit close t+1+H
  C. open  t+1 (realistic)   enter at open  t+1    -> exit close t+1+H, minus slippage

A/B use the panel's own price source (engine prices) so A reproduces the shipped
numbers. C uses the OHLC cache (data/cache/ohlc.pkl, adjusted open+close) and is
also shown with a matching OHLC close-t+1 so the venue comparison is apples-to-apples.

Verdict: if the accel top-tail's excess vs S&P survives B (and C net of slippage),
the edge is real and executable; if it collapses, the same-close entry was doing
the work. Does NOT modify edge_lib/qmodel. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import _raw_prep

SLIP_BPS = 5.0          # one-way slippage assumed on the open-entry case (on top of 10bps)


def realized(prep, ck, d, hold, mode, opens=None):
    """Realized hold-period return for one name entered under `mode`."""
    P = prep.get(ck)
    if P is None:
        return np.nan
    pos = int(np.searchsorted(P["pidx"], np.datetime64(pd.Timestamp(d), "ns"), side="right")) - 1
    arr = P["arr"]
    if mode == "same":
        a, b = pos, pos + hold
        if a < 0 or b >= len(arr):
            return np.nan
        return arr[b] / arr[a] - 1
    if mode == "c1":
        a, b = pos + 1, pos + 1 + hold
        if a < 0 or b >= len(arr):
            return np.nan
        return arr[b] / arr[a] - 1
    if mode in ("ohlc_c1", "ohlc_o1"):
        df = opens.get(ck)
        if df is None:
            return np.nan
        oi = int(df.index.searchsorted(pd.Timestamp(d), side="right")) - 1  # close t position
        a, b = oi + 1, oi + 1 + hold
        if a < 0 or b >= len(df):
            return np.nan
        entry = df["open"].iloc[a] if mode == "ohlc_o1" else df["close"].iloc[a]
        exit_ = df["close"].iloc[b]
        if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
            return np.nan
        r = exit_ / entry - 1
        if mode == "ohlc_o1":
            r -= SLIP_BPS / 1e4
        return r
    return np.nan


def book_returns(pan, holds, prep, opens, mode, hold):
    """Equal-weight per-period returns for the held books under an entry mode."""
    out = []
    for i, cks in enumerate(holds):
        d = pan.bdates[i]
        rs = [realized(prep, ck, d, hold, mode, opens) for ck in cks]
        rs = [x for x in rs if np.isfinite(x)]
        out.append(float(np.mean(rs)) if rs else 0.0)
    return np.array(out)


def net(pan, gross, turn, cost_bps=10.0):
    g = np.where(pan.ma200_on, gross, 0.25 * gross + 0.75 * pan.rf_per)
    return g - (cost_bps / 1e4) * np.asarray(turn)


def show(pan, r, label):
    ppy = pan.ppy; T = len(r)
    parts = [f"  {label:<26}"]
    for wn, yrs in (("1Y", 1), ("2Y", 2), ("5Y", 5), ("MAX", None)):
        k = T if yrs is None else min(int(round(yrs * ppy)), T)
        c, dd, sh = E.perf(r[-k:], ppy); cs, _, _ = E.perf(pan.spxf[-k:], ppy)
        parts.append(f"{wn}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}% ({(c-cs)*100:+5.1f})")
    print("  ".join(parts))


def main():
    hold = 42
    pan = E.load_edge_panel(hold=hold)
    prep = _raw_prep()
    opens = pd.read_pickle(os.path.join("data", "cache", "ohlc.pkl")) if \
        os.path.exists(os.path.join("data", "cache", "ohlc.pkl")) else {}
    print(f"hold={hold}, {pan.T} rebalances; slippage on open-entry = {SLIP_BPS}bps\n")

    s = E.EDGE_SPEC
    for n in (10, 7):
        bdates, gross, netv, turn, spxf, ndxf, holds = E._edge_full(
            hold, n, s["mcap_floor"], s["corr_cap"], s["corr_lookback"],
            s["regime_expo"], s["cost_bps"], tuple(sorted(s["signal"].items())),
            s["growth_mix"], s["growth_thresh"])
        print(f"=== n={n} (full spec) — CAGR/Sharpe/maxDD (excess vs S&P) ===")
        g_same = book_returns(pan, holds, prep, opens, "same", hold)
        g_c1 = book_returns(pan, holds, prep, opens, "c1", hold)
        show(pan, net(pan, g_same, turn), "A. same-close (baseline)")
        show(pan, net(pan, g_c1, turn), "B. close t+1 (1-day lag)")
        if opens:
            g_oc1 = book_returns(pan, holds, prep, opens, "ohlc_c1", hold)
            g_oo1 = book_returns(pan, holds, prep, opens, "ohlc_o1", hold)
            show(pan, net(pan, g_oc1, turn), "  (ohlc close t+1)")
            show(pan, net(pan, g_oo1, turn), "C. open t+1 + slippage")
        print()
    print("VERDICT: if B/C keep a clear positive excess vs S&P, the edge is executable.")


if __name__ == "__main__":
    main()
