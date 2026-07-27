"""Does the edge depend on CONTEXT — size, era, sector, regime?

Every test so far pooled the whole universe and the whole history, which
answers "does this work on average". That is the wrong question if the effect
is conditional. Momentum in particular is claimed to be alive in the current
AI-led tape while being dead on average since 2000 -- that is a testable
statement, not a vibe, and it is what this script tests.

For each factor, the top-N-minus-universe forward return is computed SEPARATELY
inside each context:

  SIZE     micro $200M-1B | small $1-2B | mid $2-10B | large $10B+
           (fundamental anomalies are documented to be strongest in small caps,
           which the $2B product floor excluded entirely)
  ERA      1999-2007 | 2008-2015 | 2016-2020 | 2021+   (the last is the AI tape)
  SECTOR   each GICS sector, so "does it work in Tech" is answerable
  REGIME   S&P above vs below its 200-day average

Read it as a hypothesis generator, not a verdict. Slicing a fixed dataset many
ways is exactly how spurious conditional effects are manufactured: with enough
buckets something always looks strong. A context result is only interesting if
the SAME context works across several factors, or the same factor works across
adjacent buckets (e.g. both 2016-2020 AND 2021+, not one alone).

Run:  .venv-mac/bin/python context_scan.py [hold]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd

import edge_data as D
import edge_lib as E

TOP_N = 10
MIN_POOL = 40          # a bucket smaller than this can't support a top-10 pick

FACTORS = {
    "ret_12_1":        +1,   # momentum, 12-1
    "ret_126":         +1,   # momentum, 6m
    "accel":           +1,   # the falsified shipped signal, for reference
    "fcf_margin":      +1,
    "gp_assets":       +1,
    "roic":            +1,
    "accruals":        -1,
    "share_issuance":  -1,
    "asset_growth":    -1,
    "shareholder_yield": +1,
    "earnings_quality": +1,
    "capex_assets":    -1,
    "fcf_yield":       +1,
}

SIZE_BUCKETS = [("micro $0.2-1B", 2e8, 1e9), ("small $1-2B", 1e9, 2e9),
                ("mid $2-10B", 2e9, 1e10), ("large $10B+", 1e10, np.inf)]
ERAS = [("1999-2007", "1999-01-01", "2008-01-01"),
        ("2008-2015", "2008-01-01", "2016-01-01"),
        ("2016-2020", "2016-01-01", "2021-01-01"),
        ("2021+ (AI)", "2021-01-01", "2099-01-01")]


def _t(a):
    a = np.asarray([v for v in a if v == v], float)
    if len(a) < 3 or a.std(ddof=1) == 0:
        return (float(a.mean()) if len(a) else 0.0), 0.0, len(a)
    return float(a.mean()), float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))), len(a)


def tails(pan, mask_fn, floor=None, ceil=None):
    """{factor: [per-rebalance top-N excess]} within the selected context."""
    out = {k: [] for k in FACTORS}
    for i, df in enumerate(pan.panels):
        if not mask_fn(i):
            continue
        d = df
        mc = pd.to_numeric(d["pit_mcap"], errors="coerce")
        if floor is not None:
            d = d[(mc >= floor) & (mc < (ceil if ceil is not None else np.inf))]
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        d = d[np.isfinite(fwd)]
        if len(d) < MIN_POOL:
            continue
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        if "fcf" in d.columns and "ev_d" in d.columns:      # units: ev is in millions
            ev = pd.to_numeric(d["ev_d"], errors="coerce") * 1e6
            d = d.assign(fcf_yield=pd.to_numeric(d["fcf"], errors="coerce") / ev.where(ev > 0))
        for col, sign in FACTORS.items():
            if col not in d.columns:
                continue
            v = pd.to_numeric(d[col], errors="coerce") * sign
            ok = np.isfinite(v)
            if ok.sum() < MIN_POOL:
                continue
            srt = fwd[ok].iloc[np.argsort(v[ok].values)]
            out[col].append(float(srt.tail(TOP_N).mean() - fwd[ok].mean()))
    return out


def table(title, blocks, pan):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    names = [b[0] for b in blocks]
    hdr = f"{'factor':<20}" + "".join(f"{n:>19}" for n in names)
    print(hdr); print("-" * len(hdr))
    res = {blk[0]: tails(pan, *blk[1:]) for blk in blocks}
    for col in FACTORS:
        cells, any_data = "", False
        for n in names:
            m, t, k = _t(res[n].get(col, []))
            if k == 0:
                cells += f"{'--':>19}"
            else:
                any_data = True
                cells += f"{m*100:>+11.2f}% t{t:>5.1f}"
        if any_data:
            print(f"{col:<20}{cells}")
    print(f"(cells: mean top-{TOP_N} excess per rebalance, and its t-stat)")


def main(argv) -> int:
    hold = int(argv[1]) if len(argv) > 1 else 63
    # load_edge_panel defaults to the top-1000 by point-in-time market cap.
    # That cut happens BEFORE any size bucketing here, so with the default the
    # micro/small buckets come back empty or near-empty and the small-cap rows
    # are meaningless. Widen the panel so small caps actually survive into it.
    universe = int(argv[2]) if len(argv) > 2 else 5000
    pan = E.load_edge_panel(hold=hold, universe=universe)
    print(f"universe cut: top {universe} by PIT market cap per rebalance")
    bd = pd.DatetimeIndex(pan.bdates)
    print(f"panel: {D.meta().get('source')} | {pan.T} rebalances "
          f"{bd[0].date()} -> {bd[-1].date()} | hold={hold}d | artifact={D.ARTIFACT_FILE}")

    table("BY SIZE (all years)",
          [(nm, (lambda i: True), lo, hi) for nm, lo, hi in SIZE_BUCKETS], pan)

    table("BY ERA (>= $2B, the product universe)",
          [(nm, (lambda a, b: lambda i: (bd[i] >= pd.Timestamp(a)) and (bd[i] < pd.Timestamp(b)))(a, b),
            2e9, None) for nm, a, b in ERAS], pan)

    table("BY ERA (small caps $0.2-2B)",
          [(nm, (lambda a, b: lambda i: (bd[i] >= pd.Timestamp(a)) and (bd[i] < pd.Timestamp(b)))(a, b),
            2e8, 2e9) for nm, a, b in ERAS], pan)

    on = np.asarray(pan.ma200_on, bool)
    table("BY REGIME (>= $2B)",
          [("S&P > 200dMA", (lambda i: bool(on[i])), 2e9, None),
           ("S&P < 200dMA", (lambda i: not bool(on[i])), 2e9, None)], pan)

    # sector: restrict the pool to one sector at a time
    secs = [s for s in pan.sectors if s and s.lower() != "unknown"][:11]
    print(f"\n{'='*100}\nBY SECTOR (>= $2B, all years) — momentum only\n{'='*100}")
    print(f"{'sector':<26}{'ret_12_1':>14}{'t':>7}{'obs':>6}")
    print("-" * 53)
    for s in secs:
        vals = []
        for i, df in enumerate(pan.panels):
            d = df[(pd.to_numeric(df["pit_mcap"], errors="coerce") >= 2e9)
                   & (df["sector"].astype(str) == s)]
            fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
            d = d[np.isfinite(fwd)]
            if len(d) < 30:
                continue
            fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
            v = pd.to_numeric(d["ret_12_1"], errors="coerce")
            ok = np.isfinite(v)
            if ok.sum() < 30:
                continue
            srt = fwd[ok].iloc[np.argsort(v[ok].values)]
            vals.append(float(srt.tail(TOP_N).mean() - fwd[ok].mean()))
        m, t, k = _t(vals)
        if k:
            print(f"{s:<26}{m*100:>+13.2f}%{t:>7.1f}{k:>6}")

    print("\nData: Sharadar SEP/SF1/DAILY point-in-time, delisted INCLUDED. "
          "Excess = top-N by factor minus the bucket mean, per rebalance. "
          "Many buckets are tested here; treat a lone strong cell as noise "
          "unless neighbouring buckets agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
