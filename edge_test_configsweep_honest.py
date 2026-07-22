"""Config leaderboard RESWEEP on the honest basis (Jared open-queue #4).

The committed edge_test_configsweep.py scores the 120-config grid on the OLD
single-sleeve / period-return / per-rebalance-regime path. This reruns the same
grid (n x mix x hold) on the CURRENT PRODUCTION basis: staggered two-sleeve,
t+1 entry, DAILY curve, CONTINUOUS 200dMA regime, net of 10bps. Reuses the
fast-path admission walks (selection is basis-independent) but computes returns
via the real daily sleeve sim.

Writes data/cache/edge_configsweep_honest.csv -- the single, consistent trial set
that edge_rigor.py's DSR and PBO both draw from (open-queue #3).

Usage: .venv/Scripts/python.exe edge_test_configsweep_honest.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import itertools, os
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as D
import edge_test_configsweep as CS   # greedy / blend / constants (selection is basis-independent)

FLOOR, THRESH = CS.FLOOR, CS.THRESH
EXPO, BPS = CS.EXPO, CS.BPS
HOLDS, MIXES, NS = CS.HOLDS, CS.MIXES, CS.NS
WINDOWS = (("MAX", None), ("20Y", 5040), ("10Y", 2520), ("5Y", 1260), ("2Y", 504), ("1Y", 252))
OUT = os.path.join("data", "cache", "edge_configsweep_honest.csv")


def _turn(sel_list):
    turn, prev = [], None
    for cks in sel_list:
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
    return np.array(turn)


def main():
    M = D.daily_return_matrix(); rd = E._ma200_daily_state()
    spx_px = D.benchmarks()["SP500"]
    rows = []
    for hold in HOLDS:
        offsets = (0, hold // 2)
        print(f"=== hold={hold}: walks for offsets {offsets} ...")
        pans, walks = {}, {}
        for off in offsets:
            pan = E.load_edge_panel(hold=hold, offset_days=off); pans[off] = pan
            wk = []
            for i, df in enumerate(pan.panels):
                d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= FLOOR]
                g = d[pd.to_numeric(d["rev_growth"], errors="coerce") >= THRESH]
                asof = pan.bdates[i]
                adm_f, ord_f = CS.greedy(d, asof, 20)
                adm_g, ord_g = CS.greedy(g, asof, 10)
                wk.append((adm_g, ord_g, adm_f, ord_f))
            walks[off] = wk
        for mix, n in itertools.product(MIXES, NS):
            sleeves = []
            for off in offsets:
                pan = pans[off]
                sel = [CS.blend(walks[off][i], n, mix) for i in range(pan.T)]
                sleeves.append(E._sleeve_daily(pan, sel, _turn(sel), M, hold, EXPO, BPS, rd))
            model = pd.concat(sleeves, axis=1).mean(axis=1, skipna=True).dropna()
            idx = model.index; r = model.to_numpy(float)
            spx = spx_px.reindex(idx, method="ffill").pct_change().fillna(0.0).to_numpy()
            T = len(r)
            for wname, days in WINDOWS:
                k = T if days is None else min(days, T)
                if k < 20:
                    continue
                c, dd, sh = E._perf_daily(r[-k:]); so = E._sortino_daily(r[-k:])
                cs, _, _ = E._perf_daily(spx[-k:])
                rows.append(dict(hold=hold, mix=mix, n=n, window=wname, cagr=c, sharpe=sh,
                                 sortino=so, dd=dd, excess=c - cs,
                                 calmar=(c / abs(dd) if dd < 0 else np.nan)))
        print(f"    hold={hold}: {len(MIXES)*len(NS)} configs")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); df.to_csv(OUT, index=False)
    print(f"\nhonest grid -> {OUT}  ({len(df)} rows)")

    # validation: the shipped product (hold=42, mix=.75, n=10, MAX) must match ~18.4/0.91/-26
    a = df[(df.hold == 42) & (df.mix == 0.75) & (df.n == 10) & (df.window == "MAX")].iloc[0]
    print(f"\nVALIDATION shipped product MAX: {a.cagr*100:.1f}% / {a.sharpe:.2f} / {a.dd*100:.0f}%  "
          f"(expect ~18.4 / 0.91 / -26)")

    mx = df[df.window == "MAX"].set_index(["hold", "mix", "n"])
    y5 = df[df.window == "5Y"].set_index(["hold", "mix", "n"])
    lb = mx.join(y5, lsuffix="_MAX", rsuffix="_5Y")
    lb["rank_score"] = (lb.sharpe_MAX.rank(pct=True) + lb.sharpe_5Y.rank(pct=True)
                        + lb.calmar_MAX.rank(pct=True) + (-lb.dd_MAX).rank(pct=True)) / 4

    def show(title, frame, k=8):
        print(f"\n-- {title} --")
        print(f"{'hold':>5}{'mix':>6}{'n':>4} | {'MAX CAGR/Sh/DD/Cal':>24} | {'5Y Sh/DD':>12} | {'exMAX':>7}")
        for (h, m, nn), r in frame.head(k).iterrows():
            print(f"{h:>5}{m:>6.2f}{nn:>4} | {r.cagr_MAX*100:6.1f}%/{r.sharpe_MAX:5.2f}/{r.dd_MAX*100:4.0f}%/"
                  f"{r.calmar_MAX:4.2f} | {r.sharpe_5Y:5.2f}/{r.dd_5Y*100:4.0f}% | {r.excess_MAX*100:+6.1f}%")
    show("TOP 8 by composite (Sharpe MAX+5Y, Calmar MAX, shallow DD)", lb.sort_values("rank_score", ascending=False))
    show("TOP 8 by MAX Sharpe", lb.sort_values("sharpe_MAX", ascending=False))
    show("TOP 8 by shallowest MAX drawdown (excess>0)", lb[lb.excess_MAX > 0].sort_values("dd_MAX", ascending=False))
    # where the shipped config ranks
    lb_sorted = lb.sort_values("rank_score", ascending=False).reset_index()
    pos = lb_sorted[(lb_sorted.hold==42)&(lb_sorted.mix==0.75)&(lb_sorted.n==10)].index
    if len(pos):
        print(f"\nShipped config (42/0.75/10) composite rank: #{int(pos[0])+1} of {len(lb_sorted)}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
