"""Is the accel signal an artifact of the UNIVERSE? IC + tail, proxy vs true-PIT.

NOT to be confused with edge_signal_ic.py, which hunts for NEW signals by
ranking many candidates on one universe. This asks the opposite question about
the ONE signal we already trade: how much of its edge survives when the
survivorship-biased top-1000 proxy is swapped for true point-in-time S&P 500
membership? Same signal, two universes.

Survivorship bias inflates LEVELS (CAGR) but largely cancels out of
CROSS-SECTIONAL rank measures -- every name in a rebalance panel shares the
same universe construction -- so rank statistics are the more bias-robust
evidence. The catch, and the reason the top-N tail is reported too: the Edge
does not trade the ranking, it trades the extreme top of it, so a ~0 broad IC
is compatible with a real edge (and vice versa).

Prints, for the proxy top-1000 and the true-PIT S&P 500 universes:
  * mean rank-IC, its t-stat, and %positive rebalances
  * decile-10 minus decile-1 forward return per rebalance
  * top-N (the traded book) minus universe-mean forward return, with t-stat

Run:  .venv-mac/bin/python edge_ic.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PIT_MEMBERSHIP_CSV",
                      str(Path(__file__).resolve().parent / "data" / "sp500_membership.csv"))

import numpy as np
from scipy import stats

import edge_lib as E

HOLD = 42
N = 10
SIGNAL = {"accel": 1.0}


def ic_report(label: str, use_pit: bool):
    E.USE_PIT_UNIVERSE = use_pit
    E.DELIST_HAIRCUT = None
    E.reset_caches()          # globals are not in any cache key -- clear ALL of them
    pan = E.load_edge_panel(hold=HOLD)
    ics, spreads, tails = [], [], []
    for df in pan.panels:
        d = df.dropna(subset=["fwd_ret"]).copy()
        d = d[np.isfinite(d["fwd_ret"])]
        if len(d) < 100:
            continue
        s = E.score(d, SIGNAL)
        ic = stats.spearmanr(s, d["fwd_ret"]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
        srt = d.assign(_s=s).sort_values("_s")
        idx = np.array_split(np.arange(len(srt)), 10)
        sp = float(np.nanmean(srt.iloc[idx[-1]]["fwd_ret"]) -
                   np.nanmean(srt.iloc[idx[0]]["fwd_ret"]))
        if np.isfinite(sp):
            spreads.append(sp)
        # what the strategy actually trades: the top-N tail vs the panel average
        tail = float(np.nanmean(srt.tail(N)["fwd_ret"]) - np.nanmean(d["fwd_ret"]))
        if np.isfinite(tail):
            tails.append(tail)
    ics = np.array(ics)
    t = float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))) if len(ics) > 1 else 0.0
    tl = np.array(tails)
    tt = float(tl.mean() / tl.std(ddof=1) * np.sqrt(len(tl))) if len(tl) > 1 else 0.0
    print(f"{label}: n={len(ics)} | IC {ics.mean():+.3f} (t={t:.1f}, "
          f"{np.mean(ics > 0)*100:.0f}%+) | D10-D1 {np.mean(spreads)*100:+.2f}%/reb | "
          f"top-{N} tail vs univ {tl.mean()*100:+.2f}%/reb (t={tt:.1f}, "
          f"{np.mean(tl > 0)*100:.0f}%+)", flush=True)


if __name__ == "__main__":
    print(f"Edge signal quality (accel, {HOLD}d clock)\n")
    ic_report("proxy top-1000     ", use_pit=False)
    ic_report("true-PIT S&P500    ", use_pit=True)
    print("\nBroad IC can be ~0 while the strategy still works: the Edge trades the"
          "\nextreme top tail, so the tail stat is the one that must survive the"
          "\ntrue-PIT universe for the signal to be considered real.")
