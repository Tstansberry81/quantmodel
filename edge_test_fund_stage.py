"""The Edge -- STAGE-2 test for the accruals earnings-quality overlay.

Stage 1 (edge_test_fundamentals.py) found ONE fundamental KPI that clears the
bar on our universe: ACCRUALS (net_margin - fcf_margin, Sloan 1996). Correct
sign at every horizon, monotonic, stable, and ORTHOGONAL to accel (rank-partial
IC -1.55%/t-2.07 at 42d). An IC of ~-1.6% is real but modest -- the question
this file answers is: does routing out / tilting away from high-accruals names
actually improve the CONCENTRATED n=7 / n=10 book, net of cost, under the honest
staggered + daily framing (the real product)?

We drive the REAL selection path (E._blend_select -> corr_cap_select -> score;
E._sleeve_daily; two staggered sleeves) so every knob matches EDGE_SPEC and the
ONLY difference between variants is the accruals overlay:
  * baseline  : accel only (reproduces the shipped product)
  * tilt w    : score = z(accel) - w * z(accruals)   (penalize high accruals)
  * gate p    : drop the worst p-fraction by accruals, THEN pick on accel

Metrics are taken on the DAILY curve (true peak-to-trough maxDD), net of the
10bps entry cost, exactly like the product. A variant only "wins" if it improves
Sharpe/DD without a return give-up that fails the DD-priority.

Usage:
    .venv/Scripts/python.exe edge_test_fund_stage.py            # hold=42 (Edge clock)
    .venv/Scripts/python.exe edge_test_fund_stage.py --hold 21
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd

import config
import edge_lib as E
import edge_data as D
from edge_test_fundamentals import attach as attach_fund

SPEC = E.EDGE_SPEC   # hold=42,n=10,mcap_floor=2e9,corr_cap=.5,corr_lookback=126,
                     # regime_expo=.25,cost_bps=10,growth_mix=.75,growth_thresh=.15,stagger


def _select_overlay(pan, panels_f, n, mode, param):
    """Per-rebalance selection through the REAL _blend_select, with an accruals
    overlay. `panels_f` = panel dfs with the `accruals` column attached."""
    holds, turn, prev = [], [], None
    for i, df in enumerate(panels_f):
        d = df[df["pit_mcap"] >= SPEC["mcap_floor"]] if SPEC["mcap_floor"] else df
        if mode == "gate" and param > 0:
            a = pd.to_numeric(d["accruals"], errors="coerce")
            keep = a <= a.quantile(1.0 - param)      # drop the worst `param` (highest accruals)
            d = d[keep | a.isna()]                    # never exclude on MISSING data (conservative)
            weights = {"accel": 1.0}
        elif mode == "tilt":
            weights = {"accel": 1.0, "accruals": -float(param)}   # high accruals -> lower score
        else:
            weights = {"accel": 1.0}
        cks = E._blend_select(d, pan.bdates[i], n, weights, SPEC["corr_cap"],
                              SPEC["corr_lookback"], SPEC["growth_mix"], SPEC["growth_thresh"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn)


def run_variant(hold, n, mode="baseline", param=0.0):
    """Daily NET return series for one variant, staggered sleeves (mirrors
    E._edge_daily), with the accruals overlay injected into selection."""
    M = D.daily_return_matrix()
    offsets = (0, hold // 2) if SPEC["stagger"] else (0,)
    series, turns = [], []
    for off in offsets:
        pan = E.load_edge_panel(hold=hold, offset_days=off)
        panels_f, _ = attach_fund(pan)
        holds, turn = _select_overlay(pan, panels_f, n, mode, param)
        series.append(E._sleeve_daily(pan, holds, turn, M, hold,
                                      SPEC["regime_expo"], SPEC["cost_bps"]))
        turns.append(float(np.mean(turn)))
    model = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
    return model, float(np.mean(turns))


def stats(model):
    r = model.to_numpy(float)
    c, dd, sh = E.perf(r, ppy=252.0)
    so = E.sortino(r, ppy=252.0)
    return c, sh, so, dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=42)
    args = ap.parse_args(); H = args.hold

    variants = [("baseline", 0.0)]
    variants += [("tilt", w) for w in (0.25, 0.5, 1.0)]
    variants += [("gate", p) for p in (0.10, 0.20, 0.30)]

    for n in (7, 10):
        print("\n" + "=" * 78)
        print(f"ACCRUALS OVERLAY -- n={n}, hold={H}, staggered/daily, net of {SPEC['cost_bps']:.0f}bps")
        print("=" * 78)
        print(f"{'variant':<16}{'CAGR':>9}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}{'turn':>8}")
        print("-" * 78)
        base = None
        for mode, param in variants:
            model, turn = run_variant(H, n, mode, param)
            c, sh, so, dd = stats(model)
            label = mode if mode == "baseline" else f"{mode} {param:g}"
            if mode == "baseline":
                base = (c, sh, so, dd)
                tag = ""
            else:
                # win = Sharpe up AND maxDD not worse (DD-priority per the campaign)
                tag = "  <-- better" if (sh > base[1] + 1e-9 and dd >= base[3] - 1e-9) else ""
            print(f"{label:<16}{c*100:>8.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%{turn*100:>7.0f}%{tag}")
        print("-" * 78)
        print("Reading it: baseline = shipped accel product. A variant only wins if Sharpe")
        print("improves WITHOUT a worse maxDD (drawdown is the campaign's priority metric).")


if __name__ == "__main__":
    main()
