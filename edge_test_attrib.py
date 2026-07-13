"""The Edge -- WHERE does the per-period return actually come from?

Motivation: accel has ~0 cross-sectional IC (edge_signal_ic.py), yet the full
product backtests at 20.7%/1.03 (hold=42, net). So the edge must live in the
non-ranking layers. Decompose each rebalance's return into:

    universe mean  ->  growth-pool mean      (the >=15% YoY rev-growth gate)
                   ->  top-N-by-accel mean   (the signal's TOP-TAIL, all it's used for)

and test each increment's mean + t-stat across all rebalances. Also:
  * conditional IC of accel INSIDE the growth pool (can it rank growers?)
  * tail curve: top 5/10/20/50-by-accel excess vs its pool, universe-wide and in-pool
  * same decomposition with the $2B floor applied (the tradeable universe)

Everything is per-period cross-sectional means -> hundreds of observations, not
a 10-name equity curve, so the t-stats mean something. No lookahead: uses the
panel's own PIT signals + fwd_ret. Does NOT modify edge_lib/qmodel. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import spearman, t_stat

GROWTH_THRESH, MCAP_FLOOR = 0.15, 2e9


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def tail_mean(df, sig, k):
    """Mean fwd_ret of the top-k names by `sig` (plain rank, no corr cap)."""
    d = df.dropna(subset=["fwd_ret"])
    s = _num(d, sig)
    top = d.loc[s.nlargest(k).index]
    return float(top["fwd_ret"].mean()) if len(top) else np.nan


def series_stats(x, label, ppy):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    ann = np.nanmean(x) * ppy * 100
    print(f"  {label:<44} mean/period={np.nanmean(x)*100:+6.2f}%  "
          f"~annualized={ann:+7.1f}%  t={t_stat(x):5.2f}  n={len(x)}")


def main():
    for hold in (42, 21):
        pan = E.load_edge_panel(hold=hold)
        ppy = pan.ppy
        print(f"\n{'='*100}\nATTRIBUTION at hold={hold}  ({pan.T} rebalances, "
              f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()})\n{'='*100}")

        gate_uni, gate_flr = [], []          # growth gate increment (no floor / with floor)
        tail_uni, tail_pool = [], []         # accel top-10 tail vs its own pool
        tails = {5: [], 10: [], 20: [], 50: []}   # tail curve within growth+floor pool
        ic_pool, ic_uni = [], []             # accel IC inside growth pool vs whole universe
        pool_sz = []
        for df in pan.panels:
            d = df.dropna(subset=["fwd_ret"])
            if len(d) < 100:
                continue
            flr = d[_num(d, "pit_mcap") >= MCAP_FLOOR]
            g_uni = d[_num(d, "rev_growth") >= GROWTH_THRESH]
            g_flr = flr[_num(flr, "rev_growth") >= GROWTH_THRESH]
            if len(g_flr) < 30:
                continue
            pool_sz.append(len(g_flr))

            uni_m, flr_m = float(d["fwd_ret"].mean()), float(flr["fwd_ret"].mean())
            gpool_m = float(g_flr["fwd_ret"].mean())
            gate_uni.append(float(g_uni["fwd_ret"].mean()) - uni_m)
            gate_flr.append(gpool_m - flr_m)

            # the signal's top-tail: top-10 by accel, universe-wide and in-pool
            tail_uni.append(tail_mean(flr, "accel", 10) - flr_m)
            tail_pool.append(tail_mean(g_flr, "accel", 10) - gpool_m)
            for k in tails:
                tails[k].append(tail_mean(g_flr, "accel", k) - gpool_m)

            # can accel rank at all, inside vs outside the pool?
            ic_uni.append(spearman(flr["accel"], flr["fwd_ret"]))
            ic_pool.append(spearman(g_flr["accel"], g_flr["fwd_ret"]))

        print(f"\n[avg growth+floor pool size: {np.mean(pool_sz):.0f} names]")
        print("\n-- LAYER 1: the >=15% rev-growth GATE (pool mean minus parent mean) --")
        series_stats(gate_uni, "gate on full universe", ppy)
        series_stats(gate_flr, "gate on $2B-floored universe (product)", ppy)

        print("\n-- LAYER 2: accel's TOP-TAIL (top-10 by accel minus its pool mean) --")
        series_stats(tail_uni, "top-10 accel in floored universe (no gate)", ppy)
        series_stats(tail_pool, "top-10 accel INSIDE growth pool (product)", ppy)

        print("\n-- tail curve inside the growth+floor pool (top-k minus pool mean) --")
        for k, v in tails.items():
            series_stats(v, f"top-{k} by accel", ppy)

        print("\n-- conditional rank-IC of accel --")
        print(f"  whole floored universe : IC={np.nanmean(ic_uni)*100:+5.2f}%  t={t_stat(np.array(ic_uni)):5.2f}")
        print(f"  inside growth pool     : IC={np.nanmean(ic_pool)*100:+5.2f}%  t={t_stat(np.array(ic_pool)):5.2f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
