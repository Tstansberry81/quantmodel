"""PEAD stage-2: does a SUE exclusion gate improve the concentrated book?

The SUE IC is sub-threshold (t~1.3) but orthogonal to accel + economically an
"earnings confirmation" filter. Test the only sensible use: route out momentum
names whose most-recent earnings surprise was BAD, before the accel pick. Real
staggered/daily path; metrics on the FULL curve and on the 2016+ coverage window
(SUE only exists from ~2016, so a full-sample read dilutes the effect).

Variants: baseline (accel) | drop SUE<0 | drop bottom-30% SUE.
Usage: .venv/Scripts/python.exe edge_test_pead_stage2.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as D
from edge_test_pead import attach as attach_sue

SPEC = E.EDGE_SPEC


def _select(pan, panels_s, n, mode, param):
    holds, turn, prev = [], [], None
    for i, df in enumerate(panels_s):
        d = df[df["pit_mcap"] >= SPEC["mcap_floor"]] if SPEC["mcap_floor"] else df
        s = pd.to_numeric(d["sue"], errors="coerce")
        if mode == "drop_neg":
            d = d[(s >= 0) | s.isna()]                       # keep positive-surprise + unknown
        elif mode == "drop_bottom":
            d = d[(s >= s.quantile(param)) | s.isna()]
        cks = E._blend_select(d, pan.bdates[i], n, {"accel": 1.0}, SPEC["corr_cap"],
                              SPEC["corr_lookback"], SPEC["growth_mix"], SPEC["growth_thresh"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn)


def build_sleeves(qeps, hold):
    out = []
    for off in (0, hold // 2):
        pan = E.load_edge_panel(hold=hold, offset_days=off)
        panels_s, _ = attach_sue(pan, qeps)
        out.append((pan, panels_s))
    return out


def run(sleeves, M, n, mode, param):
    series = []
    for pan, panels_s in sleeves:
        holds, turn = _select(pan, panels_s, n, mode, param)
        series.append(E._sleeve_daily(pan, holds, turn, M, SPEC["hold"],
                                      SPEC["regime_expo"], SPEC["cost_bps"]))
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()


def stat(model, since=None):
    m = model[model.index >= since] if since else model
    r = m.to_numpy(float); c, dd, sh = E._perf_daily(r)
    return c, sh, E._sortino_daily(r), dd


def main():
    qeps = pickle.load(open("data/cache/quarterly_eps.pkl", "rb"))
    M = D.daily_return_matrix()
    variants = [("baseline", "baseline", 0.0), ("drop SUE<0", "drop_neg", 0.0),
                ("drop bot30% SUE", "drop_bottom", 0.30)]
    for n in (7, 10):
        sleeves = build_sleeves(qeps, SPEC["hold"])
        print("\n" + "=" * 88)
        print(f"PEAD GATE -- n={n}, hold={SPEC['hold']}, staggered/daily  (FULL 2005+  |  2016+ coverage window)")
        print("=" * 88)
        print(f"{'variant':<18}{'CAGR':>7}{'Shrp':>6}{'Sort':>6}{'maxDD':>7}   "
              f"{'16+CAGR':>8}{'16+Shrp':>8}{'16+DD':>7}")
        print("-" * 88)
        base = None
        for lab, mode, param in variants:
            model = run(sleeves, M, n, mode, param)
            c, sh, so, dd = stat(model)
            c2, sh2, so2, dd2 = stat(model, since="2016-01-01")
            tag = ""
            if base is None:
                base = (sh2, dd2)
            else:
                tag = "  <-- better(16+)" if (sh2 > base[0] + 1e-9 and dd2 >= base[1] - 1e-9) else ""
            print(f"{lab:<18}{c*100:>6.1f}%{sh:>6.2f}{so:>6.2f}{dd*100:>6.0f}%   "
                  f"{c2*100:>7.1f}%{sh2:>8.2f}{dd2*100:>6.0f}%{tag}")
        print("-" * 88)


if __name__ == "__main__":
    main()
