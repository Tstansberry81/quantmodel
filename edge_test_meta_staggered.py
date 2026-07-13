"""The Edge -- #4 meta-model under the REAL product framing (staggered + daily).

The interaction meta-model beat accel in single-sleeve period returns and survived
the half-split/pool checks. Before any wiring decision, re-evaluate it the way the
product now actually runs: two staggered sleeves (offset 0 and 21d), each daily
buy-and-hold simulated, combined 50/50 -- head-to-head with the current accel
staggered-daily product over the identical (post-burn-in) window.

For each sleeve: walk-forward ElasticNetCV with interactions -> meta-label pick
(re-rank the accel top-K by the OOS score, take top-n) -> daily sim. Then combine.
Uses K=30 (best in the verdict check). Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as D
import edge_test_metamodel as MM
import edge_test_interactions as INT

K = 30
FLOOR = 2e9


def meta_holds(pan, periods, preds, n):
    """Per-rebalance meta pick (accel top-K re-ranked by OOS score), aligned to
    pan.panels; empty list for burn-in periods with no prediction."""
    holds, turn, prev = [], [], None
    for i, p in enumerate(periods):
        if p is None or preds[i] is None:
            holds.append([]); turn.append(0.0); continue
        accel, cks, score = p["accel"], p["cks"], preds[i]
        pool = np.argsort(-accel)[:K]
        order = pool[np.argsort(-score[pool])]
        sel = list(cks[order[:n]])
        cur = set(sel)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        if cur:
            prev = cur
        holds.append(sel)
    return holds, np.array(turn)


def perf_windows(r, spx, label):
    r = np.asarray(r, float); spx = np.asarray(spx, float); T = len(r)
    parts = [f"  {label:<32}"]
    for wn, k in (("1Y", 252), ("2Y", 504), ("5Y", 1260), ("MAX", None)):
        seg = r if k is None else r[-min(k, T):]
        ss = spx if k is None else spx[-min(k, T):]
        c, dd, sh = E._perf_daily(seg); cs, _, _ = E._perf_daily(ss)
        parts.append(f"{wn}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}% ({(c-cs)*100:+5.1f})")
    print("   ".join(parts))


def main():
    M = D.daily_return_matrix()
    for n in (7, 10):
        # meta sleeves (offset 0 + 21), daily-simmed, combined
        mseries = []
        for off in (0, 21):
            pan = E.load_edge_panel(hold=42, offset_days=off)
            per = INT.prep_interactions(pan)
            preds, _ = MM.walk_forward(per)
            h, t = meta_holds(pan, per, preds, n)
            mseries.append(E._sleeve_daily(pan, h, t, M, 42, 0.25, 10.0))
        meta_d = pd.concat(mseries, axis=1).mean(axis=1, skipna=True).dropna()

        # accel staggered-daily product over the same dates
        idx, acc, spx, ndx, _, _ = E._edge_daily(
            42, n, FLOOR, 0.5, 126, 0.25, 10.0, (("accel", 1.0),), 0.75, 0.15, True)
        acc_s = pd.Series(acc, index=idx); spx_s = pd.Series(spx, index=idx)
        common = meta_d.index.intersection(acc_s.index)
        m = meta_d.reindex(common).to_numpy()
        a = acc_s.reindex(common).to_numpy()
        sp = spx_s.reindex(common).to_numpy()
        print(f"\n{'='*104}\nn={n}  staggered + daily, common window "
              f"{common[0].date()} -> {common[-1].date()} ({len(common)} days)\n{'='*104}")
        perf_windows(a, sp, "accel product (current)")
        perf_windows(m, sp, f"meta-model K={K} (interactions)")
    print("\nDONE.  Apples-to-apples vs the live staggered/daily product.")


if __name__ == "__main__":
    main()
