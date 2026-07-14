"""Consolidated backtest report: product vs every candidate upgrade from the campaign.

Runs the tradeable Edge (staggered/daily, net of cost) for the product and each
validated/candidate overlay, and reports CAGR, Sharpe, Sortino, maxDD, and alpha
(excess CAGR vs S&P 500 TR) over MAX / 10Y / 5Y for n=7 and n=10.

Variants:
  product          accel, per-rebalance 200dMA@0.25 (the shipped rule)
  +continuous      accel, 200dMA evaluated DAILY (round-3 robust-DD win)
  +panic           accel, Daniel-Moskowitz panic-state de-risk (round-4 Sharpe win)
  +gp_gate         GP/assets quality gate (drop bottom 40%), product regime
  +upgraded        GP gate + continuous regime (the paper-upgraded sleeve)

Reuses the real product path (E._edge_daily / _sleeve_daily) + the campaign
harnesses. Authoritative product windows also cross-checked via run_edge_backtest.
Usage: .venv/Scripts/python.exe edge_backtest_report.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd

import edge_lib as E
import edge_data as D
from edge_test_timing import book_daily, market_exposures, RF_D
from edge_test_fund_stage2 import build_sleeves
from edge_test_combine import _select

SPEC = E.EDGE_SPEC
HOLD = SPEC["hold"]
WINDOWS = (("MAX", None), ("10Y", 2520), ("5Y", 1260))


def overlay(pure, e):
    e = e.reindex(pure.index).ffill().fillna(1.0)
    de = e.diff().abs().fillna(0.0)
    return pd.Series(e.values * pure.values + (1 - e.values) * RF_D - 1e-4 * de.values,
                     index=pure.index)


def gated_daily(sleeves, M, n, regime_expo, gate=0.40):
    series = []
    for pan, panels_f in sleeves:
        holds, turn = _select(pan, panels_f, n, "static_gate", gate)
        series.append(E._sleeve_daily(pan, holds, turn, M, HOLD, regime_expo, SPEC["cost_bps"]))
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()


def stats_over(model, spx, k):
    """CAGR / Sharpe / Sortino / maxDD / alpha over the last k days (None = all)."""
    r = model.to_numpy(float); s = spx.to_numpy(float)
    if k is not None:
        r = r[-k:]; s = s[-k:]
    c, dd, sh = E._perf_daily(r)
    so = E._sortino_daily(r)
    cs, _, _ = E._perf_daily(s)
    return c, sh, so, dd, c - cs


def spx_for(idx):
    return D.benchmarks()["SP500"].reindex(idx, method="ffill").pct_change().fillna(0.0)


def main():
    with open("data/cache/fund_canonical.pkl", "rb") as f:
        fund = pickle.load(f)
    M = D.daily_return_matrix()

    for n in (10, 7):
        print("\n" + "=" * 92)
        print(f"EDGE BACKTEST -- n={n}, hold={HOLD}, staggered/daily, net of {SPEC['cost_bps']:.0f}bps")
        print("=" * 92)

        # build the daily series for every variant
        prod, spx = book_daily(n, SPEC["regime_expo"])
        pure, _ = book_daily(n, 1.0)
        exps = market_exposures(pure.index)
        cont = overlay(pure, exps["d200_25"])
        panic = overlay(pure, exps["panic"])
        sleeves = build_sleeves(fund, HOLD)
        gp_gate = gated_daily(sleeves, M, n, SPEC["regime_expo"])         # GP gate + product regime
        gp_pure = gated_daily(sleeves, M, n, 1.0)                         # GP gate, no de-risk
        exps_g = market_exposures(gp_pure.index)
        upgraded = overlay(gp_pure, exps_g["d200_25"])                    # GP gate + continuous regime

        variants = [("product", prod), ("+continuous", cont), ("+panic", panic),
                    ("+gp_gate", gp_gate), ("+upgraded", upgraded)]

        for wn, k in WINDOWS:
            print(f"\n  --- {wn} window ---")
            print(f"  {'variant':<14}{'CAGR':>8}{'alpha':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}")
            # S&P reference
            sp_arr = spx.to_numpy()[-k:] if k else spx.to_numpy()
            cs, dds, shs = E._perf_daily(sp_arr)
            print(f"  {'S&P 500 TR':<14}{cs*100:>7.1f}%{'—':>8}{shs:>8.2f}{'—':>9}{dds*100:>7.0f}%")
            for lab, series in variants:
                sp = spx_for(series.index)
                c, sh, so, dd, a = stats_over(series, sp, k)
                print(f"  {lab:<14}{c*100:>7.1f}%{a*100:>+7.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%")

    # authoritative product cross-check (windows w/ excess) from the shipped path
    print("\n" + "=" * 92)
    print("CROSS-CHECK: run_edge_backtest (shipped path) -- product n=10, per-window excess vs S&P")
    print("=" * 92)
    bt = E.run_edge_backtest("MAX")
    if bt.get("ok"):
        print(f"  period {bt['period'][0]} -> {bt['period'][1]}, {bt['n_rebalances']} rebalances")
        print(f"  {'window':<7}{'CAGR':>8}{'S&P':>8}{'excess':>9}{'Sharpe':>8}{'maxDD':>8}")
        for w in bt["windows"]:
            print(f"  {w['w']:<7}{w['cagr']*100:>7.1f}%{w['sp_cagr']*100:>7.1f}%"
                  f"{w['excess']*100:>+8.1f}%{w['sharpe']:>8.2f}{w['dd']*100:>7.0f}%")


if __name__ == "__main__":
    main()
