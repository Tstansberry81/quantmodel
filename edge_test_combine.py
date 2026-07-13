"""The Edge -- SIGNAL-COMBINATION / REGIME-CONDITIONAL test (round-4 methods).

Tests the verified round-4 methods that act on the SELECTION layer, on the real
staggered/daily product path (only the selection rule differs from EDGE_SPEC):

  #3 COMBINATION -- equal-weight rank composite of accel + GP/assets
     (Stambaugh-Yuan 2017): score = z(accel) + w*z(gp_assets). w=1 -> equal
     z-composite; w=0.5 -> accel-heavy. The robust, low-parameter answer to why
     the elastic-net meta-model failed (no noisy return-forecast coefficients).

  #2 REGIME-CONDITIONING -- apply the GP/assets quality gate ONLY when the market
     is stressed (S&P below its 200dMA), relaxed in calm up-regimes
     (Asness-Frazzini-Pedersen 2019: quality is defensive, pays off in bad states).
     vs the STATIC gate (last session's winner) which gates in all regimes.

Reference rows: baseline (accel only) and static GP gate 0.40 (prior winner).
Book fixed otherwise; metrics on the daily curve. Needs data/cache/fund_canonical.pkl.

Usage: .venv/Scripts/python.exe edge_test_combine.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd

import edge_lib as E
import edge_data as D
from edge_test_fund_stage2 import build_sleeves

SPEC = E.EDGE_SPEC


def _select(pan, panels_f, n, mode, param):
    holds, turn, prev = [], [], None
    for i, df in enumerate(panels_f):
        d = df[df["pit_mcap"] >= SPEC["mcap_floor"]] if SPEC["mcap_floor"] else df
        weights = {"accel": 1.0}
        if mode == "composite":
            weights = {"accel": 1.0, "gp_assets": float(param)}
        elif mode == "static_gate":
            a = pd.to_numeric(d["gp_assets"], errors="coerce")
            d = d[(a >= a.quantile(param)) | a.isna()]
        elif mode == "regime_gate":
            # gate ONLY in stressed regime (S&P below 200dMA); relax in calm
            if not bool(pan.ma200_on[i]):
                a = pd.to_numeric(d["gp_assets"], errors="coerce")
                d = d[(a >= a.quantile(param)) | a.isna()]
        cks = E._blend_select(d, pan.bdates[i], n, weights, SPEC["corr_cap"],
                              SPEC["corr_lookback"], SPEC["growth_mix"], SPEC["growth_thresh"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn)


def run(sleeves, M, hold, n, mode, param):
    series = []
    for pan, panels_f in sleeves:
        holds, turn = _select(pan, panels_f, n, mode, param)
        series.append(E._sleeve_daily(pan, holds, turn, M, hold,
                                      SPEC["regime_expo"], SPEC["cost_bps"]))
    model = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
    r = model.to_numpy(float)
    c, dd, sh = E._perf_daily(r)
    return c, sh, E.sortino(r, ppy=252.0), dd


def main():
    with open("data/cache/fund_canonical.pkl", "rb") as f:
        fund = pickle.load(f)
    M = D.daily_return_matrix()
    variants = [
        ("baseline (accel)",      "baseline",    0.0),
        ("composite w=1.0",       "composite",   1.0),   # #3 equal z-composite
        ("composite w=0.5",       "composite",   0.5),   # #3 accel-heavy
        ("static gate 0.40",      "static_gate", 0.40),  # prior winner (all regimes)
        ("regime gate 0.40",      "regime_gate", 0.40),  # #2 gate only when stressed
        ("regime gate 0.50",      "regime_gate", 0.50),
    ]
    for n in (7, 10):
        print("\n" + "=" * 74)
        print(f"COMBINATION / REGIME-CONDITIONAL -- n={n}, hold={SPEC['hold']}, staggered/daily")
        print("=" * 74)
        print(f"{'variant':<20}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}")
        print("-" * 74)
        sleeves = build_sleeves(fund, SPEC["hold"])
        base = None
        for lab, mode, param in variants:
            c, sh, so, dd = run(sleeves, M, SPEC["hold"], n, mode, param)
            if base is None:
                base = (c, sh, so, dd); tag = ""
            else:
                tag = "  <-- better" if (sh > base[1] + 1e-9 and dd >= base[3] - 1e-9) else ""
            print(f"{lab:<20}{c*100:>7.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%{tag}")
        print("-" * 74)
        print("Better = Sharpe up AND maxDD no worse vs baseline (drawdown priority).")


if __name__ == "__main__":
    main()
