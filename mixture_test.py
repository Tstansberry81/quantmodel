"""Composite fundamental factors: mixtures, breadth, clock — with a FRESH holdout.

Implements every recommendation from the single-factor battery at once:

  fundamentals  drop P/E (dead at all 11 thresholds); FCF yield (FCF/EV) as the
                value leg instead; FCF margin + FCF growth as the core; ROIC/ROE
                as a high gate not a ranker; add Novy-Marx gross profitability,
                Sloan accruals and net share issuance.
  quant         composite z-scores rather than single factors; SECTOR-RELATIVE
                ranking (an absolute gross-margin cut just sorts by industry);
                breadth 10 -> 100; slower clocks (fundamentals update quarterly,
                so a 42-day rebalance churns on unchanged information);
                deflated-Sharpe haircut for the number of trials.

THREE-WAY SPLIT. Everything before has used 1999-2012 to choose and 2012-2026
to check, so 2012-2026 is no longer clean. Here:

    TRAIN   1999-2012   free to look at
    TEST    2012-2020   used for the earlier work, reported for continuity
    HOLDOUT 2021+       untouched until this script ran

The holdout column is the only one that has not been contaminated by earlier
searching, so it is the one that counts.

Run:  .venv-mac/bin/python mixture_test.py
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd

import edge_data as D
import edge_lib as E

MCAP_FLOOR = 2e9
HOLDOUT_START = pd.Timestamp("2021-01-01")
TEST_START = pd.Timestamp("2012-09-01")

# (column, sign) -- sign already oriented so that HIGHER = better.
LEGS = {
    "fcf_margin":      +1,   # survived both halves, OOS t=2.2
    "growth_fcf_1y":   +1,   # survived both halves
    "roic":            +1,
    "roe":             +1,
    "gp_assets":       +1,   # Novy-Marx gross profitability
    "accruals":        -1,   # Sloan: earnings not backed by cash are bad
    "share_issuance":  -1,   # dilution bad, buybacks good
    "fcf_yield":       +1,   # FCF/EV -- replaces the dead P/E leg
    "ret_12_1":        +1,   # momentum, for the mixed composite
}

MIXES = {
    "FCF core (margin+growth)":   {"fcf_margin": 1, "growth_fcf_1y": 1},
    "Quality (ROIC+ROE+GP)":      {"roic": 1, "roe": 1, "gp_assets": 1},
    "Novy-Marx GP only":          {"gp_assets": 1},
    "Accruals+issuance (clean)":  {"accruals": 1, "share_issuance": 1},
    "FCF yield (value leg)":      {"fcf_yield": 1},
    "FCF core + quality":         {"fcf_margin": 1, "growth_fcf_1y": 1,
                                   "roic": 1, "gp_assets": 1},
    "Everything fundamental":     {"fcf_margin": 1, "growth_fcf_1y": 1, "roic": 1,
                                   "gp_assets": 1, "accruals": 1,
                                   "share_issuance": 1, "fcf_yield": 1},
    "Fundamental + momentum":     {"fcf_margin": 1, "growth_fcf_1y": 1, "roic": 1,
                                   "gp_assets": 1, "ret_12_1": 1},
    "Quality + momentum":         {"gp_assets": 1, "roic": 1, "ret_12_1": 1},
}


def _z(s: pd.Series) -> pd.Series:
    """Winsorised cross-sectional z-score. Fundamental ratios have violent
    outliers (a near-zero denominator gives ROIC in the thousands); clipping at
    the 1/99th percentile stops one name dominating the composite."""
    v = pd.to_numeric(s, errors="coerce")
    lo, hi = v.quantile(0.01), v.quantile(0.99)
    v = v.clip(lo, hi)
    sd = v.std()
    return (v - v.mean()) / sd if sd and sd > 0 else pd.Series(0.0, index=s.index)


def composite(d: pd.DataFrame, mix: dict, sector_rel: bool) -> pd.Series:
    """Weighted sum of z-scored legs. `sector_rel` z-scores WITHIN sector, so a
    software name competes on margin against software, not against retail."""
    tot = pd.Series(0.0, index=d.index)
    wsum = 0.0
    for col, w in mix.items():
        if col not in d.columns:
            continue
        raw = pd.to_numeric(d[col], errors="coerce") * LEGS.get(col, 1)
        if sector_rel and "sector" in d.columns:
            z = raw.groupby(d["sector"].astype(str)).transform(_z)
        else:
            z = _z(raw)
        tot = tot + w * z.fillna(0.0)
        wsum += abs(w)
    return tot / (wsum or 1.0)


def evaluate(pan, mix, n, sector_rel, gate):
    """Mean excess forward return per rebalance, split TRAIN/TEST/HOLDOUT."""
    buckets = {"train": [], "test": [], "holdout": []}
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= MCAP_FLOOR]
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        d = d[np.isfinite(fwd)]
        if len(d) < 100:
            continue
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        # FCF yield = FCF / enterprise value, using the DAILY (month-end PIT)
        # EV rather than market cap: EV charges the buyer for the debt, which
        # is the whole point of preferring it to P/E. Negative EV is nonsense
        # and is dropped rather than inverted into a huge fake yield.
        # UNITS: SF1 reports fcf in dollars (median ~$7.4e8) while DAILY reports
        # ev in MILLIONS (median ~2.4e4 = $24bn). Dividing them raw inflates the
        # yield by 1e6 -- a leg that still ranks, still runs, and is silently
        # meaningless. Scale EV to dollars first.
        if "fcf" in d.columns and "ev_d" in d.columns:
            ev = pd.to_numeric(d["ev_d"], errors="coerce") * 1e6
            d = d.assign(fcf_yield=pd.to_numeric(d["fcf"], errors="coerce")
                         / ev.where(ev > 0))
        if gate:                       # ROIC/ROE as a GATE, not a ranker
            g = pd.to_numeric(d.get("roic"), errors="coerce")
            keep = (g > gate).fillna(False)
            if keep.sum() < max(n, 30):
                continue
            d, fwd = d[keep], fwd[keep]
        sc = composite(d, mix, sector_rel)
        top = fwd.iloc[np.argsort(sc.values)].tail(n)
        ex = float(top.mean() - fwd.mean())
        dt = pan.bdates[i]
        key = "holdout" if dt >= HOLDOUT_START else ("test" if dt >= TEST_START else "train")
        buckets[key].append(ex)
    return buckets


def _stat(a):
    a = np.asarray([v for v in a if v == v], float)
    if len(a) < 3 or a.std(ddof=1) == 0:
        return 0.0, 0.0, len(a)
    return float(a.mean()), float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))), len(a)


def deflated_t(t: float, n_trials: int) -> float:
    """Multiple-testing haircut: the expected max |t| of `n_trials` independent
    noise draws is ~sqrt(2 ln N). A t below that is what pure search produces."""
    return float(t - np.sqrt(2.0 * np.log(max(n_trials, 2))))


def main() -> int:
    print(f"panel source: {D.meta().get('source')}")
    trials = 0
    results = []
    for hold in (42, 126, 252):
        pan = E.load_edge_panel(hold=hold)
        print(f"\n{'='*104}\nCLOCK {hold} trading days  |  {pan.T} rebalances  "
              f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n{'='*104}")
        hdr = (f"{'mixture':<28}{'n':>4}{'sec':>5}{'gate':>6}"
               f"{'TRAIN':>9}{'t':>6}{'TEST':>9}{'t':>6}{'HOLDOUT':>10}{'t':>6}{'obs':>5}")
        print(hdr); print("-" * len(hdr))
        rows = []
        for name, mix in MIXES.items():
            for n in (10, 50, 100):
                for sector_rel in (False, True):
                    for gate in (None, 0.20):
                        b = evaluate(pan, mix, n, sector_rel, gate)
                        tr, tr_t, _ = _stat(b["train"])
                        te, te_t, _ = _stat(b["test"])
                        ho, ho_t, ho_n = _stat(b["holdout"])
                        trials += 1
                        rows.append((ho, name, n, sector_rel, gate, tr, tr_t,
                                     te, te_t, ho, ho_t, ho_n))
        for (_, name, n, sr, gate, tr, tr_t, te, te_t, ho, ho_t, ho_n) in \
                sorted(rows, key=lambda r: -r[0])[:12]:
            print(f"{name:<28}{n:>4}{'Y' if sr else '-':>5}"
                  f"{('>%.0f%%' % (gate*100)) if gate else '-':>6}"
                  f"{tr*100:>+8.2f}%{tr_t:>6.1f}{te*100:>+8.2f}%{te_t:>6.1f}"
                  f"{ho*100:>+9.2f}%{ho_t:>6.1f}{ho_n:>5}")
        results.extend(rows)

    best = max(results, key=lambda r: r[0])
    print(f"\n{'='*104}")
    print(f"{trials} configurations tested. Deflated-t threshold for that many "
          f"trials: {np.sqrt(2*np.log(trials)):.2f}")
    print(f"Best holdout: {best[1]} n={best[2]} sector={'Y' if best[3] else 'N'} "
          f"gate={best[4]} -> {best[0]*100:+.2f}%/reb (t={best[10]:.1f}, "
          f"deflated t={deflated_t(best[10], trials):+.1f})")
    print("A deflated t below 0 means the result is indistinguishable from the "
          "best of that many noise draws.")
    print("\nData: Sharadar SEP/SF1/DAILY point-in-time (ART, datekey, +1 day "
          "filing lag; multiples month-end PIT), delisted INCLUDED; universe "
          "top-1000 by PIT market cap >= $2B. Excess = mean forward return of "
          "the top-n composite names minus the universe mean, per rebalance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
