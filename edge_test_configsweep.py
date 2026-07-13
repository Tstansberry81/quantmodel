"""The Edge -- full configuration sweep over the FOUR UI dials.

Grid: n in {5..10} x growth-mix in {0,25,50,75,100} x hold in {21,42,63,126}
= 120 configs, each scored over every window (1Y/2Y/5Y/10Y/20Y/MAX). All other
spec values fixed at the product (floor $2B, corr-cap 0.50, regime 25%, 10bps).

FAST PATH: corr-cap greedy admission has a prefix property -- the first k
admissions don't depend on the basket-size target, and the pool walks don't
depend on mix. So per (hold, rebalance) we compute TWO admission walks (full
pool to depth 20, growth pool to depth 10) and compose all 30 (mix, n) baskets
from them, ~10x faster than 120 independent runs. Validated below against the
web app's known anchors before any results are trusted.

Ranking guard-rail: the window is a REPORTING lens, not a model parameter --
ranking by a config's prettiest window is recency-mining. Leaderboards rank on
the long windows (MAX / 5Y; Sharpe, Calmar, maxDD per the user's DD priority);
per-window numbers are reported for context. Writes the full grid to CSV in the
scratchpad. Does NOT modify edge_lib/qmodel. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import itertools
import os
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as TB   # self-contained Edge data layer (was tech_bias_lib; has daily_return_matrix)

W = {"accel": 1.0}
FLOOR, CAP, CLB = 2e9, 0.50, 126
EXPO, BPS, THRESH = 0.25, 10.0, 0.15
HOLDS = (21, 42, 63, 126)
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)
NS = (5, 6, 7, 8, 9, 10)
WINDOWS = ("1Y", "2Y", "5Y", "10Y", "20Y", "MAX")
OUT_CSV = os.path.join(
    os.environ.get("EDGE_SWEEP_OUT", "."), "edge_configsweep.csv")


def greedy(d, asof, depth):
    """Replicates corr_cap_select's admission walk to `depth`, returning
    (admitted, full score order). Prefix property: admitted[:k] == the walk
    that corr_cap_select(n=k) would admit."""
    dd = d.dropna(subset=["fwd_ret"]).copy()
    dd["_c"] = E.score(dd, W)
    order = list(dd.sort_values("_c", ascending=False)["company_key"])
    M = TB.daily_return_matrix()
    win = M[M.index <= asof].tail(CLB)
    adm = []
    for ck in order:
        if len(adm) >= depth:
            break
        if ck not in win.columns:
            continue
        if not adm:
            adm.append(ck)
            continue
        c = win[[ck] + adm].corr().iloc[0, 1:].abs().max()
        if pd.isna(c) or c <= CAP:
            adm.append(ck)
    return adm, order


def compose(adm, order, k):
    """corr_cap_select(n=k) == admitted prefix + backfill by score order."""
    sel = list(adm[:k])
    if len(sel) < k:
        for ck in order:
            if ck not in sel:
                sel.append(ck)
            if len(sel) >= k:
                break
    return sel[:k]


def blend(walk, n, mix):
    """Replicates E._blend_select from the two cached walks."""
    adm_g, ord_g, adm_f, ord_f = walk
    K = int(round(mix * n))
    gpick = compose(adm_g, ord_g, K) if K > 0 else []
    sel = list(gpick)
    for ck in compose(adm_f, ord_f, n + len(gpick)):
        if len(sel) >= n:
            break
        if ck not in sel:
            sel.append(ck)
    return sel[:n]


def main():
    rows = []
    for hold in HOLDS:
        print(f"\n=== hold={hold}: building panel + admission walks ...")
        pan = E.load_edge_panel(hold=hold)
        ppy = pan.ppy
        walks = []
        for i, df in enumerate(pan.panels):
            d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= FLOOR]
            g = d[pd.to_numeric(d["rev_growth"], errors="coerce") >= THRESH]
            asof = pan.bdates[i]
            adm_f, ord_f = greedy(d, asof, 20)
            adm_g, ord_g = greedy(g, asof, 10)
            walks.append((adm_g, ord_g, adm_f, ord_f))
            if (i + 1) % 40 == 0:
                print(f"    walks {i+1}/{pan.T}")

        for mix, n in itertools.product(MIXES, NS):
            sel_list = [blend(walks[i], n, mix) for i in range(pan.T)]
            turn, prev = [], None
            for cks in sel_list:
                cur = set(cks)
                turn.append(1 - len(cur & prev) / len(cur) if prev and cur
                            else (1.0 if cur else 0.0))
                prev = cur
            gross = E.period_returns(pan, sel_list)
            gross = np.where(pan.ma200_on, gross,
                             EXPO * gross + (1 - EXPO) * pan.rf_per)
            net = gross - (BPS / 1e4) * np.array(turn)
            T = len(net)
            for wname in WINDOWS:
                yrs = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "20Y": 20}.get(wname)
                k = T if yrs is None else min(int(round(yrs * ppy)), T)
                if k < 2:
                    continue
                c, dd, sh = E.perf(net[-k:], ppy)
                cs, _, _ = E.perf(pan.spxf[-k:], ppy)
                rows.append(dict(hold=hold, mix=mix, n=n, window=wname,
                                 cagr=c, sharpe=sh, dd=dd, sp_cagr=cs,
                                 excess=c - cs,
                                 calmar=(c / abs(dd) if dd < 0 else np.nan),
                                 turnover=float(np.mean(turn))))
        print(f"    hold={hold}: {len(MIXES)*len(NS)} configs scored")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nfull grid written: {OUT_CSV}  ({len(df)} rows)")

    # ---- validation anchors (must match the web app / campaign numbers) ----
    print("\nVALIDATION vs known anchors (hold=42, mix=75):")
    for n, want in ((10, "20.7%/1.03/-21%"), (7, "25.9%/1.13/-25%")):
        a = df[(df.hold == 42) & (df.mix == 0.75) & (df.n == n) & (df.window == "MAX")].iloc[0]
        print(f"  n={n}: got {a.cagr*100:.1f}%/{a.sharpe:.2f}/{a.dd*100:.0f}%   expected {want}")

    # ---- leaderboards on the long windows ----
    mx = df[df.window == "MAX"].set_index(["hold", "mix", "n"])
    y5 = df[df.window == "5Y"].set_index(["hold", "mix", "n"])
    lb = mx.join(y5, lsuffix="_MAX", rsuffix="_5Y")
    lb["avg_sharpe"] = (lb.sharpe_MAX + lb.sharpe_5Y) / 2
    lb["rank_score"] = (lb.sharpe_MAX.rank(pct=True) + lb.sharpe_5Y.rank(pct=True)
                        + lb.calmar_MAX.rank(pct=True) + (-lb.dd_MAX).rank(pct=True)) / 4

    def show(title, frame, k=10):
        print(f"\n-- {title} --")
        print(f"{'hold':>5}{'mix':>6}{'n':>4} | {'MAX: CAGR/Sh/DD/Calmar':>26} | "
              f"{'5Y: CAGR/Sh/DD':>20} | {'exc MAX':>8}{'turn':>6}")
        for (h, m, n), r in frame.head(k).iterrows():
            print(f"{h:>5}{m:>6.2f}{n:>4} | {r.cagr_MAX*100:7.1f}%/{r.sharpe_MAX:5.2f}/"
                  f"{r.dd_MAX*100:4.0f}%/{r.calmar_MAX:5.2f} | "
                  f"{r.cagr_5Y*100:6.1f}%/{r.sharpe_5Y:5.2f}/{r.dd_5Y*100:4.0f}% | "
                  f"{r.excess_MAX*100:+7.1f}%{r.turnover_MAX:6.2f}")

    show("TOP 10 by composite rank (Sharpe MAX + Sharpe 5Y + Calmar MAX + shallow DD)",
         lb.sort_values("rank_score", ascending=False))
    show("TOP 10 by MAX Sharpe", lb.sort_values("sharpe_MAX", ascending=False))
    show("TOP 10 by MAX Calmar (CAGR per unit of drawdown)",
         lb.sort_values("calmar_MAX", ascending=False))
    show("TOP 10 by shallowest MAX drawdown (excess>0 only)",
         lb[lb.excess_MAX > 0].sort_values("dd_MAX", ascending=False))
    show("TOP 10 by 5Y Sharpe", lb.sort_values("sharpe_5Y", ascending=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
