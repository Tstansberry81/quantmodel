"""Fundamental screen battery on survivorship-free data, IN vs OUT of sample.

Two different questions, kept separate because they answer different things:

  RANK  -- does the factor sort the cross-section? (rank-IC, and the top-decile
           minus bottom-decile forward return). This is the factor-investing
           question and is hard to overfit: ~1,000 names x ~160 rebalances.

  SCREEN-- does a THRESHOLD beat the universe? ("P/E <= 14", "ROIC > 20%").
           This is the analyst's question and is what an equity screen actually
           does. Reported as the mean forward return of everything passing the
           screen minus the universe mean, plus how many names survive -- a
           screen that passes 3 names isn't a strategy, it's an anecdote.

EVERY number is printed for the in-sample half AND the out-of-sample half.
With this many tests some will look good on one half by luck; the OOS column
is there so that is immediately visible instead of quietly believed. Read the
two together -- a screen that works in-sample and dies out-of-sample is noise,
and there will be several.

Run:  .venv-mac/bin/python fundamental_scan.py
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd
from scipy import stats

import edge_data as D
import edge_lib as E

HOLD, MCAP_FLOOR, MIN_NAMES = 42, 2e9, 5

# ---- continuous factors: which way is "good"? ------------------------------
RANK_FACTORS = {
    "roic":            +1, "roe":             +1, "roa":            +1,
    "grossmargin":     +1, "netmargin":       +1, "fcf_margin":     +1,
    "ebitda_margin":   +1, "growth_fcf_1y":   +1, "growth_netinc_1y": +1,
    "growth_revenue_1y": +1,
    "pe_d":            -1, "pb_d":            -1, "ps_d":           -1,
    "evebitda_d":      -1,                      # cheap = good, so sign flips
    "de":              -1, "debt_ebitda":     -1, "currentratio":   +1,
}

# ---- threshold screens -----------------------------------------------------
SCREENS: list[tuple[str, str, callable]] = []
for _x in range(10, 21):                      # P/E 10..20, integers, as asked
    SCREENS.append((f"P/E <= {_x}", "pe_d", (lambda c: lambda s: (s > 0) & (s <= c))(_x)))
for _x in (0.05, 0.10, 0.15, 0.20, 0.25):
    SCREENS.append((f"FCF margin > {_x:.0%}", "fcf_margin", (lambda c: lambda s: s > c)(_x)))
for _x in (0.10, 0.15, 0.20, 0.25, 0.30):
    SCREENS.append((f"ROIC > {_x:.0%}", "roic", (lambda c: lambda s: s > c)(_x)))
for _x in (0.15, 0.20, 0.25, 0.30):
    SCREENS.append((f"ROE > {_x:.0%}", "roe", (lambda c: lambda s: s > c)(_x)))
for _x in (0.30, 0.40, 0.50, 0.60):
    SCREENS.append((f"Gross margin > {_x:.0%}", "grossmargin", (lambda c: lambda s: s > c)(_x)))
for _x in (0.0, 0.10, 0.20):
    SCREENS.append((f"FCF growth > {_x:.0%}", "growth_fcf_1y", (lambda c: lambda s: s > c)(_x)))
for _x in (0.5, 1.0, 2.0):
    SCREENS.append((f"Debt/EBITDA < {_x}", "debt_ebitda", (lambda c: lambda s: (s >= 0) & (s < c))(_x)))
for _x in (0.5, 1.0):
    SCREENS.append((f"Debt/Equity < {_x}", "de", (lambda c: lambda s: (s >= 0) & (s < c))(_x)))


def _t(a):
    a = np.asarray([v for v in a if v == v], float)
    if len(a) < 3 or a.std(ddof=1) == 0:
        return (float(a.mean()) if len(a) else 0.0), 0.0, len(a)
    return float(a.mean()), float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))), len(a)


def run(pan, lo, hi):
    """Return {name: (rank_stats | screen_stats)} over rebalances [lo, hi)."""
    ranks: dict[str, list] = {k: [] for k in RANK_FACTORS}
    decs: dict[str, list] = {k: [] for k in RANK_FACTORS}
    scr: dict[str, list] = {n: [] for n, _, _ in SCREENS}
    cnt: dict[str, list] = {n: [] for n, _, _ in SCREENS}
    for i in range(lo, hi):
        df = pan.panels[i]
        d = df[df["pit_mcap"] >= MCAP_FLOOR]
        d = d[np.isfinite(pd.to_numeric(d["fwd_ret"], errors="coerce"))]
        if len(d) < 100:
            continue
        fwd = pd.to_numeric(d["fwd_ret"], errors="coerce")
        univ = float(fwd.mean())
        for col, sign in RANK_FACTORS.items():
            if col not in d.columns:
                continue
            v = pd.to_numeric(d[col], errors="coerce") * sign
            ok = np.isfinite(v)
            if ok.sum() < 100:
                continue
            ic = stats.spearmanr(v[ok], fwd[ok]).statistic
            if np.isfinite(ic):
                ranks[col].append(float(ic))
            srt = fwd[ok].iloc[np.argsort(v[ok].values)]
            idx = np.array_split(np.arange(len(srt)), 10)
            decs[col].append(float(srt.iloc[idx[-1]].mean() - srt.iloc[idx[0]].mean()))
        for name, col, pred in SCREENS:
            if col not in d.columns:
                continue
            v = pd.to_numeric(d[col], errors="coerce")
            sel = fwd[pred(v).fillna(False)]
            if len(sel) >= MIN_NAMES:
                scr[name].append(float(sel.mean() - univ))
                cnt[name].append(len(sel))
    return ranks, decs, scr, cnt


def main() -> int:
    pan = E.load_edge_panel(hold=HOLD)
    split = pan.T // 2
    print(f"panel: {D.meta().get('source')} | {pan.T} rebalances "
          f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()}")
    print(f"IN: -> {pan.bdates[split-1].date()}   OUT: {pan.bdates[split].date()} ->"
          f"   (>= ${MCAP_FLOOR/1e9:.0f}B, {HOLD}d hold, forward returns net of nothing)\n")
    r_is, d_is, s_is, c_is = run(pan, 0, split)
    r_oo, d_oo, s_oo, c_oo = run(pan, split, pan.T)

    print("=" * 92)
    print("RANK FACTORS — does it sort the cross-section? (sign already oriented so + = good)")
    print("=" * 92)
    h = f"{'factor':<20}{'IS IC':>8}{'t':>6}{'IS D10-D1':>12}   {'OOS IC':>8}{'t':>6}{'OOS D10-D1':>12}"
    print(h); print("-" * len(h))
    rows = []
    for k in RANK_FACTORS:
        im, it, n = _t(r_is.get(k, []))
        om, ot, _ = _t(r_oo.get(k, []))
        if n == 0:
            continue
        dim = np.mean(d_is[k]) if d_is[k] else float("nan")
        dom = np.mean(d_oo[k]) if d_oo[k] else float("nan")
        rows.append((om, k, im, it, dim, om, ot, dom))
    for _, k, im, it, dim, om, ot, dom in sorted(rows, reverse=True):
        print(f"{k:<20}{im:>+8.3f}{it:>6.1f}{dim*100:>+11.2f}%   {om:>+8.3f}{ot:>6.1f}{dom*100:>+11.2f}%")

    print("\n" + "=" * 92)
    print("THRESHOLD SCREENS — passing names' forward return minus the universe, per rebalance")
    print("=" * 92)
    h2 = f"{'screen':<24}{'IS xs':>9}{'t':>6}{'IS n':>7}   {'OOS xs':>9}{'t':>6}{'OOS n':>7}"
    print(h2); print("-" * len(h2))
    srows = []
    for name, _, _ in SCREENS:
        im, it, n = _t(s_is.get(name, []))
        om, ot, _ = _t(s_oo.get(name, []))
        if n == 0:
            continue
        srows.append((om, name, im, it, np.mean(c_is[name]) if c_is[name] else 0,
                      om, ot, np.mean(c_oo[name]) if c_oo[name] else 0))
    for _, name, im, it, ni, om, ot, no in sorted(srows, reverse=True):
        print(f"{name:<24}{im*100:>+8.2f}%{it:>6.1f}{ni:>7.0f}   "
              f"{om*100:>+8.2f}%{ot:>6.1f}{no:>7.0f}")

    print(f"\n{len(RANK_FACTORS)} rank factors + {len(SCREENS)} screens = "
          f"{len(RANK_FACTORS)+len(SCREENS)} tests. At that count roughly one in "
          "twenty crosses t=2 by chance alone, so treat any single result as "
          "noise unless BOTH halves agree.")
    print("Data: Sharadar SEP/SF1/DAILY point-in-time (ART, datekey, +1 trading "
          "day filing lag; multiples month-end PIT), delisted INCLUDED; universe "
          f"top-1000 by PIT market cap >= ${MCAP_FLOOR/1e9:.0f}B; rebalance {HOLD} trading days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
