"""Drawdown-reduction research harness.

WHY THESE THREE LEVERS. Decomposing the shipped model's drawdowns showed the two
worst non-COVID episodes (2021-08 -39.8%, 2000-03 -38.3%) happened with the S&P
above its 200dMA 95-97% of the time, while the index itself fell only ~10%. With
portfolio beta at 0.82, those are NOT market-exposure events -- they are momentum
unwinds. So the market regime gate cannot fix them by construction, and the
levers worth testing are ones that see something the gate cannot:

  vol_target   scale exposure by the BOOK's own trailing vol (momentum vol spikes
               before momentum crashes) -- Moreira-Muir volatility management
  name_trend   require each holding to be above its OWN 200dMA (the 2021 names
               broke their own trend months before the index broke its)
  cont_regime  evaluate the market gate daily instead of freezing it at rebalance
               (a control: targets COVID, the one fast beta crash)

Metrics come from the PRODUCTION daily curve (RESEARCH_RULES #1). _selfcheck()
asserts this file's metric functions reproduce run_edge_backtest's headline
numbers on the baseline before any variant is believed.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

import config
import edge_data as D
import edge_lib as E

RF_D = config.RISK_FREE_ANNUAL / 252.0


def daily_series(**over):
    """Production daily NET model returns for a spec override."""
    s = {**E.EDGE_SPEC, **over}
    idx, model, spx, ndx, avg_to, prim = E._edge_daily(
        s["hold"], s["n"], s["mcap_floor"], s["corr_cap"], s["corr_lookback"],
        s["regime_expo"], s["cost_bps"], tuple(sorted(s["signal"].items())),
        s["growth_mix"], s["growth_thresh"], s["stagger"], s["continuous_regime"],
        s["fcf_screen"], s["sector_cap"], s.get("name_trend", False))
    return pd.DatetimeIndex(idx), np.asarray(model, float), np.asarray(spx, float)


def vol_target(r, target=0.20, lookback=63, cap=1.0, cost_bps=10.0):
    """Scale exposure to a constant target vol; uninvested cash earns the
    risk-free rate (same convention as the regime gate).

    CAUSAL: the scale applied to day t is computed from returns through t-1
    only. Using day t's own vol would be a look-ahead that makes any vol target
    look brilliant -- it would be de-risking on days it already knows are bad.

    Re-levering is a trade, so the change in scale is charged turnover at the
    same cost as a rebalance. Ignoring that would flatter a lever whose whole
    mechanism is trading more often.
    """
    s = pd.Series(r)
    rv = s.rolling(lookback).std().shift(1) * np.sqrt(252)   # shift(1) = causal
    scale = (target / rv).clip(upper=cap).fillna(cap).to_numpy()
    scaled = scale * r + (1.0 - scale) * RF_D
    relever = np.abs(np.diff(scale, prepend=scale[0]))
    return scaled - (cost_bps / 1e4) * relever, scale


def metrics(idx, r, bench=None):
    """Delegates to production's _perf_daily rather than reimplementing it.

    The first version of this reimplemented Sharpe as (ann.return - rf)/vol and
    disagreed with the site by 0.15. Production defines it as
    sqrt(252)*mean/std with NO risk-free subtraction, and annualizes CAGR at
    252/len rather than calendar years. Whether that's the better definition is
    a separate argument; a research harness that quietly uses a different one
    produces numbers nobody can reconcile against the page.
    """
    cagr, dd, sharpe = E._perf_daily(r)
    out = {"cagr": cagr, "vol": float(np.nanstd(r) * np.sqrt(252)),
           "sharpe": sharpe, "maxdd": dd,
           "mar": cagr / abs(dd) if dd < 0 else np.nan}
    if bench is not None:
        out["excess"] = cagr - E._perf_daily(bench)[0]
    return out


EPISODES = [("COVID", "2020-02-19", "2020-07-07"),
            ("2021 unwind", "2021-08-09", "2024-02-05"),
            ("dot-com", "2000-03-03", "2005-08-01"),
            ("2015-16", "2015-07-17", "2017-11-06"),
            ("2025", "2025-02-14", "2025-07-25")]


def episode_dd(idx, r, start, end):
    """Max drawdown WITHIN a window, measured from the peak at the window start
    (not the all-time peak), so episodes are comparable across variants."""
    w = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    if w.sum() < 5:
        return np.nan
    eq = np.cumprod(1.0 + r[w])
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def _selfcheck(idx, r):
    """RESEARCH_RULES #1: this file's numbers must equal the production path's."""
    prod = E.run_edge_backtest(window="MAX")["performance"]["model"]
    mine = metrics(idx, r)
    for k, pk in (("cagr", "cagr"), ("sharpe", "sharpe"), ("maxdd", "max_drawdown")):
        if abs(mine[k] - prod[pk]) > 5e-4:
            raise SystemExit(f"SELF-CHECK FAILED {k}: harness {mine[k]:.6f} vs "
                             f"production {prod[pk]:.6f} -- fix before trusting anything")
    print(f"self-check OK: CAGR {mine['cagr']:.2%}  Sharpe {mine['sharpe']:.3f}  "
          f"maxDD {mine['maxdd']:.2%} match production\n")


def row(name, idx, r, bench, extra=""):
    m = metrics(idx, r, bench)
    eps = "  ".join(f"{episode_dd(idx, r, a, b):6.1%}" for _, a, b in EPISODES)
    print(f"{name:<26} {m['cagr']:7.2%} {m['sharpe']:6.2f} {m['maxdd']:7.1%} "
          f"{m['mar']:5.2f} | {eps} {extra}")
    return m


def main():
    idx, base, spx = daily_series()
    _selfcheck(idx, base)

    hdr = (f"{'variant':<26} {'CAGR':>7} {'Sharpe':>6} {'maxDD':>7} {'MAR':>5} | "
           + "  ".join(f"{n:>6}" for n, _, _ in EPISODES))
    print(hdr); print("-" * len(hdr))
    row("baseline (shipped)", idx, base, spx)

    results = {}
    # --- lever 1: vol targeting (grid) -------------------------------------
    for tgt, lb in itertools.product((0.15, 0.20, 0.25), (21, 63)):
        r, sc = vol_target(base, target=tgt, lookback=lb)
        results[f"vt{tgt:.2f}_{lb}"] = row(
            f"volTarget {tgt:.0%} / {lb}d", idx, r, spx,
            extra=f"avg expo {sc.mean():.0%}")

    # --- lever 2: name-level trend filter ----------------------------------
    idx2, nt, _ = daily_series(name_trend=True)
    results["nt"] = row("nameTrend (own 200dMA)", idx2, nt, spx)

    # --- lever 3: continuous regime (control) ------------------------------
    idx3, cr, _ = daily_series(continuous_regime=True)
    results["cr"] = row("contRegime (daily gate)", idx3, cr, spx)

    # --- combinations ------------------------------------------------------
    print()
    idx4, ntcr, _ = daily_series(name_trend=True, continuous_regime=True)
    row("nameTrend + contRegime", idx4, ntcr, spx)
    for tgt in (0.15, 0.20):
        r, _ = vol_target(nt, target=tgt, lookback=63)
        row(f"nameTrend + vt{tgt:.0%}", idx2, r, spx)
        r2, _ = vol_target(ntcr, target=tgt, lookback=63)
        row(f"all three (vt{tgt:.0%})", idx4, r2, spx)

    # --- walk-forward: does it survive out of sample? -----------------------
    print("\nWALK-FORWARD  (train 1999-2012 / test 2013-2026, same params)")
    print(f"{'variant':<26} {'trainCAGR':>9} {'trainMAR':>8} {'testCAGR':>9} {'testMAR':>8}")
    cut = pd.Timestamp("2013-01-01")
    cand = {"baseline": (idx, base), "nameTrend": (idx2, nt),
            "contRegime": (idx3, cr),
            "volTarget 20%/63d": (idx, vol_target(base, 0.20, 63)[0]),
            "volTarget 15%/63d": (idx, vol_target(base, 0.15, 63)[0]),
            "nameTrend + vt20%": (idx2, vol_target(nt, 0.20, 63)[0]),
            "all three (vt20%)": (idx4, vol_target(ntcr, 0.20, 63)[0])}
    for nm, (ix, rr) in cand.items():
        tr, te = ix < cut, ix >= cut
        a, b = metrics(ix[tr], rr[tr]), metrics(ix[te], rr[te])
        print(f"{nm:<26} {a['cagr']:8.2%} {a['mar']:8.2f} {b['cagr']:8.2%} {b['mar']:8.2f}")

    n_tests = 6 + 1 + 1 + 5
    print(f"\nN tests this run = {n_tests}; multiple-testing t-threshold "
          f"sqrt(2*ln N) = {np.sqrt(2*np.log(n_tests)):.2f}")


if __name__ == "__main__":
    main()
