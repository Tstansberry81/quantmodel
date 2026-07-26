"""Which signals actually rank forward returns on survivorship-free data?

The Edge's acceleration signal looked strong on the fiscal.ai panel and is
NEGATIVE on Sharadar. Before tuning any portfolio, the prior question has to be
answered on honest data: does the signal rank the cross-section at all, and is
there a better one already sitting in the panel?

Rank-IC over ~1,000 names x ~160 rebalances is far harder to overfit than a
10-stock CAGR over a few overlapping windows, so discovery happens HERE, not in
a backtest. For every signal we report:

  IC        mean Spearman(signal, forward return), its t-stat, %positive
  D10-D1    top-decile minus bottom-decile forward return per rebalance
  tail      what the strategy actually trades: mean forward return of the top-N
            by signal, minus the universe mean, with a t-stat
  H1 / H2   the same tail statistic on each half of the sample -- a signal that
            only works in one half is a regime artifact, not an edge

Nothing here fits parameters. Every number is a plain descriptive statistic of
one fixed column, so there is no search to overfit with.

Run:  .venv-mac/bin/python signal_scan.py            # Sharadar panel, 42d
      EDGE_ARTIFACT=backtest_data.fiscal.pkl .venv-mac/bin/python signal_scan.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")   # hold the universe rule fixed

import numpy as np
import pandas as pd
from scipy import stats

import edge_data as D
import edge_lib as E

HOLD = 42
TOP_N = 10
MCAP_FLOOR = 2e9

# Signals already carried in the panel. `vol_21` is inverted (low-vol is the
# documented effect), `hi_252` is distance-from-high (higher = nearer the high).
SIGNALS = {
    "accel      (3m - prior 3m)": ("accel", +1),
    "ret_63     (3m momentum)":   ("ret_63", +1),
    "ret_126    (6m momentum)":   ("ret_126", +1),
    "ret_12_1   (12-1 momentum)": ("ret_12_1", +1),
    "ret_21     (1m reversal)":   ("ret_21", -1),
    "hi_252     (52w high prox)": ("hi_252", +1),
    "rs_63      (rel strength)":  ("rs_63", +1),
    "vol_21     (LOW vol)":       ("vol_21", -1),
    "rev_growth (YoY sales)":     ("rev_growth", +1),
}


def _stats(vals):
    a = np.asarray([v for v in vals if v == v], float)
    if len(a) < 3:
        return 0.0, 0.0, 0.0
    t = float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))) if a.std(ddof=1) > 0 else 0.0
    return float(a.mean()), t, float(np.mean(a > 0))


def scan(hold: int = HOLD):
    pan = E.load_edge_panel(hold=hold)
    src = D.meta().get("source", "fiscal")
    print(f"panel: {src} | {pan.T} rebalances | {pan.bdates[0].date()} -> "
          f"{pan.bdates[-1].date()} | artifact={D.ARTIFACT_FILE}\n")

    half = pan.T // 2
    rows = []
    for label, (col, sign) in SIGNALS.items():
        ics, spreads, tails, tails_h1, tails_h2 = [], [], [], [], []
        for i, df in enumerate(pan.panels):
            d = df[df["pit_mcap"] >= MCAP_FLOOR]
            d = d.dropna(subset=["fwd_ret", col])
            d = d[np.isfinite(d["fwd_ret"]) & np.isfinite(d[col])]
            if len(d) < 100:
                continue
            s = sign * d[col].astype(float)
            fwd = d["fwd_ret"].astype(float)
            ic = stats.spearmanr(s, fwd).statistic
            if np.isfinite(ic):
                ics.append(float(ic))
            srt = fwd.iloc[np.argsort(s.values)]           # ascending by signal
            idx = np.array_split(np.arange(len(srt)), 10)
            spreads.append(float(srt.iloc[idx[-1]].mean() - srt.iloc[idx[0]].mean()))
            tail = float(srt.tail(TOP_N).mean() - fwd.mean())
            tails.append(tail)
            (tails_h1 if i < half else tails_h2).append(tail)

        ic_m, ic_t, ic_p = _stats(ics)
        tl_m, tl_t, tl_p = _stats(tails)
        h1 = _stats(tails_h1)[0]
        h2 = _stats(tails_h2)[0]
        rows.append((label, ic_m, ic_t, ic_p, float(np.mean(spreads)) if spreads else 0.0,
                     tl_m, tl_t, tl_p, h1, h2))

    hdr = (f"{'signal':<28}{'IC':>7}{'t':>6}{'%+':>5}  {'D10-D1':>8}"
           f"{'tail':>8}{'t':>6}{'%+':>5}  {'tailH1':>8}{'tailH2':>8}")
    print(hdr); print("-" * len(hdr))
    for (lab, icm, ict, icp, sp, tlm, tlt, tlp, h1, h2) in sorted(rows, key=lambda r: -r[5]):
        print(f"{lab:<28}{icm:>+7.3f}{ict:>6.1f}{icp*100:>4.0f}%  {sp*100:>+7.2f}%"
              f"{tlm*100:>+7.2f}%{tlt:>6.1f}{tlp*100:>4.0f}%  {h1*100:>+7.2f}%{h2*100:>+7.2f}%")
    print(f"\ntail = mean forward return of the top-{TOP_N} by signal minus the "
          f"universe mean, per rebalance ({hold}d hold, >= ${MCAP_FLOOR/1e9:.0f}B).")
    print("A signal is only interesting if the tail is positive, its t-stat is "
          "meaningful, AND both halves agree.")
    return rows


if __name__ == "__main__":
    scan(int(sys.argv[1]) if len(sys.argv) > 1 else HOLD)
