"""Why does production disagree with the research runs? Decompose it.

Two questions, answered with measurements rather than argument:

  1. Is production actually running on the full survivorship-free dataset --
     delisted names included, dead companies buyable and holdable into their
     delisting? If the panel silently dropped them, every number is wrong.

  2. Where does the research-vs-production gap come from? Four suspects, each
     isolated: the universe cut, the correlation cap, staggered sleeves, and
     the measurement basis (per-rebalance sampling vs the daily curve).

Run:  .venv-mac/bin/python diagnose_gap.py
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
import pandas as pd

import edge_data as D
import edge_lib as E

FLOOR = 1e10


def survivorship_audit():
    data = D.load_bt_data()
    dead = {ck for ck, b in data.items()
            if (b.get("meta") or {}).get("trading_status") != "Active"}
    print(f"artifact: {len(data):,} names, {len(dead):,} delisted "
          f"({len(dead)/len(data)*100:.0f}%)  source={D.meta().get('source')}")

    for uni in (1000, 20000):
        pan = E.load_edge_panel(hold=42, universe=uni)
        rows = sum(len(p) for p in pan.panels)
        dead_rows = sum(int(p["company_key"].isin(dead).sum()) for p in pan.panels)
        elig = [int((pd.to_numeric(p["pit_mcap"], errors="coerce") >= FLOOR).sum())
                for p in pan.panels]
        dead_elig = sum(int(((pd.to_numeric(p["pit_mcap"], errors="coerce") >= FLOOR)
                             & p["company_key"].isin(dead)).sum()) for p in pan.panels)
        print(f"\nuniverse={uni:<6} panel rows {rows:>9,}  of which delisted "
              f"{dead_rows:>8,} ({dead_rows/max(rows,1)*100:.0f}%)")
        print(f"{'':14} eligible >= ${FLOOR/1e9:.0f}B per rebalance: "
              f"median {int(np.median(elig))}, min {min(elig)}, max {max(elig)}"
              f"  | delisted among them: {dead_elig:,}")
    return dead


def picked_dead(dead):
    """Does the SHIPPED book ever actually hold a company that later dies?"""
    pan = E.load_edge_panel(hold=42, universe=20000)
    n_dead_picks, n_picks = 0, 0
    for i, df in enumerate(pan.panels):
        d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= FLOOR]
        if "fcf_margin" in d.columns:
            m = pd.to_numeric(d["fcf_margin"], errors="coerce")
            k = (m >= m.median()).fillna(False)
            if k.sum() >= 20:
                d = d[k]
        v = pd.to_numeric(d.get("ret_12_1"), errors="coerce")
        d = d[np.isfinite(v)]
        if len(d) < 10:
            continue
        pick = d.iloc[np.argsort(pd.to_numeric(d["ret_12_1"], errors="coerce").values)].tail(10)
        n_picks += len(pick)
        n_dead_picks += int(pick["company_key"].isin(dead).sum())
    print(f"\nshipped book held {n_dead_picks:,} of {n_picks:,} positions "
          f"({n_dead_picks/max(n_picks,1)*100:.1f}%) in companies that LATER DELISTED")


def gap():
    base = dict(E.EDGE_SPEC)
    print(f"\n{'configuration':<44}{'CAGR':>8}{'excess':>9}{'Sharpe':>8}{'maxDD':>8}")
    print("-" * 77)
    for lbl, over, uni in (
            ("production as shipped (universe=1000)", {}, 1000),
            ("  + full universe (20000)", {}, 20000),
            ("  + no corr cap", {"corr_cap": None}, 20000),
            ("  + no stagger", {"corr_cap": None, "stagger": False}, 20000),
            ("  + per-rebalance regime", {"corr_cap": None, "stagger": False,
                                          "continuous_regime": False}, 20000),
    ):
        E.load_edge_panel.__wrapped__ if False else None
        spec = {**base, **over, "universe": uni}
        # run_edge_backtest doesn't take universe; patch the default it reads
        _orig = E.UNIVERSE
        E.UNIVERSE = uni
        E.reset_caches()
        bt = E.run_edge_backtest(window="MAX", spec={k: v for k, v in spec.items()
                                                    if k != "universe"})
        E.UNIVERSE = _orig
        if not bt.get("ok"):
            print(f"{lbl:<44}  FAILED {bt.get('reason')}"); continue
        m, s = bt["performance"]["model"], bt["performance"]["sp500"]
        print(f"{lbl:<44}{m['cagr']*100:>7.1f}%{(m['cagr']-s['cagr'])*100:>+8.1f}%"
              f"{m['sharpe']:>8.2f}{m['max_drawdown']*100:>7.0f}%")
    print("\nAll rows are the DAILY-curve basis (true intra-period peak-to-trough).")


if __name__ == "__main__":
    dead = survivorship_audit()
    picked_dead(dead)
    gap()
