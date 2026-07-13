"""The Edge -- STAGE-2 for CANONICAL fundamental overlays (general signal + direction).

Generalizes edge_test_fund_stage.py to any canonical KPI from fund_canonical.pkl.
Stage-1 (edge_test_fund_canonical.py) surfaced two quality signals that ADD to
accel with incremental IC t>2: gross profitability (GP/assets, Novy-Marx) and the
full 9-signal Piotroski F-Score. Both are POSITIVE (high = good). This asks the
only question that matters for the product: does overlaying them improve the
concentrated n=7 / n=10 book, net of cost, under the honest staggered/daily frame?

Drives the REAL path (E._blend_select -> corr_cap_select -> score; E._sleeve_daily;
staggered sleeves) so only the overlay differs from EDGE_SPEC:
  * baseline : accel only (reproduces the shipped product)
  * tilt w   : score = z(accel) + dir * w * z(signal)
  * gate p   : keep the best (1-p) fraction by signal, THEN pick on accel

`dir` = +1 for POSITIVE signals (gp_assets, fscore) -> favor/keep HIGH values;
        -1 for NEGATIVE signals (accruals, share_iss) -> favor/keep LOW values.

Usage:
    .venv/Scripts/python.exe edge_test_fund_stage2.py --signal gp_assets --dir 1
    .venv/Scripts/python.exe edge_test_fund_stage2.py --signal fscore --dir 1 --hold 42
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import argparse, pickle
import numpy as np, pandas as pd

import edge_lib as E
import edge_data as D
from edge_test_fund_canonical import attach as attach_canon

SPEC = E.EDGE_SPEC


def _select_overlay(pan, panels_f, n, sig, direction, mode, param):
    holds, turn, prev = [], [], None
    for i, df in enumerate(panels_f):
        d = df[df["pit_mcap"] >= SPEC["mcap_floor"]] if SPEC["mcap_floor"] else df
        if mode == "gate" and param > 0:
            a = pd.to_numeric(d[sig], errors="coerce")
            if direction > 0:                              # keep HIGH quality: drop bottom p
                keep = a >= a.quantile(param)
            else:                                          # keep LOW: drop top p
                keep = a <= a.quantile(1.0 - param)
            d = d[keep | a.isna()]                          # never exclude on missing data
            weights = {"accel": 1.0}
        elif mode == "tilt":
            weights = {"accel": 1.0, sig: direction * float(param)}
        else:
            weights = {"accel": 1.0}
        cks = E._blend_select(d, pan.bdates[i], n, weights, SPEC["corr_cap"],
                              SPEC["corr_lookback"], SPEC["growth_mix"], SPEC["growth_thresh"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn)


def build_sleeves(fund, hold):
    """Attach canonical fundamentals ONCE per staggered sleeve (expensive); the
    overlay only changes selection, so this is reused across every variant."""
    offsets = (0, hold // 2) if SPEC["stagger"] else (0,)
    sleeves = []
    for off in offsets:
        pan = E.load_edge_panel(hold=hold, offset_days=off)
        panels_f, _ = attach_canon(pan, fund)
        sleeves.append((pan, panels_f))
    return sleeves


def run_variant(sleeves, M, hold, n, sig, direction, mode="baseline", param=0.0):
    series = []
    for pan, panels_f in sleeves:
        holds, turn = _select_overlay(pan, panels_f, n, sig, direction, mode, param)
        series.append(E._sleeve_daily(pan, holds, turn, M, hold,
                                      SPEC["regime_expo"], SPEC["cost_bps"]))
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()


def stats(model):
    r = model.to_numpy(float)
    c, dd, sh = E.perf(r, ppy=252.0)
    return c, sh, E.sortino(r, ppy=252.0), dd


def _half_stats(model):
    """(full, 1st-half, 2nd-half) each = (CAGR, Sharpe, maxDD) on the daily curve."""
    r = model.to_numpy(float); h = len(r) // 2
    def p(x):
        c, dd, sh = E.perf(x, ppy=252.0); return c, sh, dd
    return p(r), p(r[:h]), p(r[h:])


def robust(sleeves, M, hold, sig, direction, fracs):
    """Finer gate sweep + half-split -- a real overlay shows a SMOOTH plateau and
    holds in BOTH halves; a single winning cell is luck."""
    for n in (7, 10):
        print("\n" + "=" * 92)
        print(f"ROBUSTNESS -- {sig} gate (dir={direction:+d}), n={n}, hold={hold}, staggered/daily")
        print("=" * 92)
        print(f"{'variant':<14}{'CAGR':>8}{'Shrp':>6}{'DD':>6}   "
              f"{'1H CAGR':>8}{'1H Shrp':>8}{'1H DD':>7}   {'2H CAGR':>8}{'2H Shrp':>8}{'2H DD':>7}")
        print("-" * 92)
        for mode, param in [("baseline", 0.0)] + [("gate", f) for f in fracs]:
            model = run_variant(sleeves, M, hold, n, sig, direction, mode, param)
            (fc, fs, fdd), (c1, s1, dd1), (c2, s2, dd2) = _half_stats(model)
            lab = "baseline" if mode == "baseline" else f"gate {param:g}"
            print(f"{lab:<14}{fc*100:>7.1f}%{fs:>6.2f}{fdd*100:>5.0f}%   "
                  f"{c1*100:>7.1f}%{s1:>8.2f}{dd1*100:>6.0f}%   {c2*100:>7.1f}%{s2:>8.2f}{dd2*100:>6.0f}%")
        print("-" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", required=True)
    ap.add_argument("--dir", type=int, default=1, choices=(1, -1))
    ap.add_argument("--hold", type=int, default=42)
    ap.add_argument("--pkl", default="data/cache/fund_canonical.pkl")
    ap.add_argument("--robust", action="store_true",
                    help="finer gate sweep + half-split stability instead of the tilt/gate grid")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        fund = pickle.load(f)

    if args.robust:
        M = D.daily_return_matrix()
        print(f"Attaching {args.signal} to staggered sleeves (once) ...")
        sleeves = build_sleeves(fund, args.hold)
        robust(sleeves, M, args.hold, args.signal, args.dir,
               fracs=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50))
        return

    variants = [("baseline", 0.0)]
    variants += [("tilt", w) for w in (0.25, 0.5, 1.0)]
    variants += [("gate", p) for p in (0.10, 0.20, 0.30)]

    print(f"Attaching {args.signal} to staggered sleeves (once) ...")
    M = D.daily_return_matrix()
    sleeves = build_sleeves(fund, args.hold)

    for n in (7, 10):
        print("\n" + "=" * 80)
        print(f"{args.signal} overlay (dir={args.dir:+d}) -- n={n}, hold={args.hold}, "
              f"staggered/daily, net {SPEC['cost_bps']:.0f}bps")
        print("=" * 80)
        print(f"{'variant':<16}{'CAGR':>9}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}")
        print("-" * 80)
        base = None
        for mode, param in variants:
            model = run_variant(sleeves, M, args.hold, n, args.signal, args.dir, mode, param)
            c, sh, so, dd = stats(model)
            label = mode if mode == "baseline" else f"{mode} {param:g}"
            if mode == "baseline":
                base = (c, sh, so, dd); tag = ""
            else:
                tag = "  <-- better" if (sh > base[1] + 1e-9 and dd >= base[3] - 1e-9) else ""
            print(f"{label:<16}{c*100:>8.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%{tag}")
        print("-" * 80)
        print("A variant wins only if Sharpe improves WITHOUT a worse maxDD (DD-priority).")


if __name__ == "__main__":
    main()
