"""edge_test_corrcap.py -- validate the CORRELATION CAP overlay for The Edge.

Question: does capping pairwise return correlation in the basket (no two holdings
with |corr| > cap) roughly HALVE realized portfolio correlation and lift
risk-adjusted return vs the plain top-N momentum basket?

Plain `select_topN` vs `corr_cap_select` at caps {0.4, 0.5, 0.6}, n in {10, 15}.
We measure realized avg pairwise |corr| per basket (trailing 126d ending at each
rebalance, point-in-time), performance over 1Y/2Y/5Y/MAX, and selection depth
(how far down the ranked list the cap forces us to reach).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import tech_bias_lib as TB

WEIGHTS = {"ret_63": 1.0}          # the product default (3-month momentum)
CAPS = [0.4, 0.5, 0.6]
NS = [10, 15]
LOOKBACK = 126


def avg_pair_corr(M, cks, end_loc, lookback=LOOKBACK):
    """Mean off-diagonal absolute pairwise correlation for `cks` over the
    trailing `lookback` rows ending at calendar position end_loc (point in time)."""
    cks = [c for c in cks if c in M.columns]
    if len(cks) < 2:
        return np.nan
    lo = max(0, end_loc - lookback)
    win = M.iloc[lo:end_loc][cks]
    C = win.corr().abs().values
    iu = np.triu_indices_from(C, k=1)
    vals = C[iu]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def selection_depth(df, chosen, weights):
    """Rank (1-based) of the LAST admitted name in the score-ordered list.
    For a plain top-N this equals n; for a capped basket it's larger when the
    cap forces skips. Returns depth and #skipped."""
    d = df.dropna(subset=["fwd_ret"]).copy()
    d["_c"] = E.score(d, weights)
    order = list(d.sort_values("_c", ascending=False)["company_key"])
    pos = [order.index(c) + 1 for c in chosen if c in order]
    if not pos:
        return np.nan, np.nan
    depth = max(pos)
    return depth, depth - len(chosen)


def build(pan, n, mode, cap=None):
    """Return holdings list-of-lists for every rebalance."""
    holds = []
    for df in pan.panels:
        if mode == "plain":
            holds.append(E.select_topN(df, n=n, weights=WEIGHTS))
        else:
            holds.append(E.corr_cap_select(df, n=n, weights=WEIGHTS, cap=cap, lookback=LOOKBACK))
    return holds


def measure_corr_and_depth(pan, M, cal, holds, weights):
    """Avg realized pairwise |corr| across rebalances + avg selection depth/skips."""
    corrs, depths, skips = [], [], []
    for i, d in enumerate(pan.bdates):
        end_loc = int(cal.searchsorted(d, side="right"))   # corr measured on data up TO rebalance
        corrs.append(avg_pair_corr(M, holds[i], end_loc))
        dep, sk = selection_depth(pan.panels[i], holds[i], weights)
        depths.append(dep); skips.append(sk)
    return (np.nanmean(corrs), np.nanmean(depths), np.nanmean(skips))


def windows_row(pan, rets, windows=(("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None))):
    out = {}
    rets = np.asarray(rets, float)
    for wname, k in windows:
        kk = len(rets) if k is None else min(k, len(rets))
        c, dd, sh = E.perf(rets[-kk:])
        out[wname] = (c, sh, dd)
    return out


def main():
    print("Loading Edge panel (~1-2 min)...")
    pan = E.load_edge_panel()
    M = TB.daily_return_matrix()
    cal = M.index
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n")

    # ----- assemble all variants -----
    variants = []   # (label, n, mode, cap)
    for n in NS:
        variants.append((f"plain   n={n}", n, "plain", None))
        for cap in CAPS:
            variants.append((f"cap{cap} n={n}", n, "cap", cap))

    results = {}
    for label, n, mode, cap in variants:
        holds = build(pan, n, mode, cap)
        rets = E.period_returns(pan, holds)
        avgc, depth, skip = measure_corr_and_depth(pan, M, cal, holds, WEIGHTS)
        win = windows_row(pan, rets)
        results[label] = dict(n=n, mode=mode, cap=cap, avgc=avgc,
                              depth=depth, skip=skip, win=win)

    # baseline S&P for context
    spx_win = windows_row(pan, pan.spxf)

    # ===== (a) AVERAGE CORRELATION =====
    print("=" * 74)
    print("(a) REALIZED AVG PAIRWISE |CORR|  (trailing 126d, mean over rebalances)")
    print("=" * 74)
    for n in NS:
        plain_c = results[f"plain   n={n}"]["avgc"]
        print(f"\n  n={n}:  plain = {plain_c:.3f}")
        for cap in CAPS:
            r = results[f"cap{cap} n={n}"]
            ratio = r["avgc"] / plain_c if plain_c else np.nan
            print(f"         cap={cap}: {r['avgc']:.3f}   "
                  f"({ratio*100:.0f}% of plain, {(1-ratio)*100:+.0f}% reduction)")

    # ===== (b) PERFORMANCE TABLE =====
    print("\n" + "=" * 74)
    print("(b) PERFORMANCE  (CAGR / Sharpe / maxDD) per window")
    print("=" * 74)
    hdr = f"  {'variant':<14}"
    for w in ("1Y", "2Y", "5Y", "MAX"):
        hdr += f"|{w:^22}"
    print(hdr)
    print(f"  {'':<14}" + "|" + " CAGR   Shrp   DD     " * 4)
    print("  " + "-" * 102)
    for label, n, mode, cap in variants:
        r = results[label]
        line = f"  {label:<14}"
        for w in ("1Y", "2Y", "5Y", "MAX"):
            c, sh, dd = r["win"][w]
            line += f"|{c*100:6.1f}%{sh:6.2f}{dd*100:6.0f}% "
        print(line)
        if label.startswith("plain") and n == NS[-1]:
            pass
    # S&P reference
    line = f"  {'S&P 500':<14}"
    for w in ("1Y", "2Y", "5Y", "MAX"):
        c, sh, dd = spx_win[w]
        line += f"|{c*100:6.1f}%{sh:6.2f}{dd*100:6.0f}% "
    print(line)

    # ===== selection depth =====
    print("\n" + "=" * 74)
    print("(c) SELECTION DEPTH  (avg rank of last admitted name; avg #skipped)")
    print("=" * 74)
    for label, n, mode, cap in variants:
        r = results[label]
        print(f"  {label:<14} depth={r['depth']:5.1f}  skipped={r['skip']:4.1f}  "
              f"(picks {n} from top {r['depth']:.0f})")

    # ===== summary deltas vs plain (Sharpe & DD) =====
    print("\n" + "=" * 74)
    print("    SHARPE / maxDD DELTA vs plain (same n), full-window MAX")
    print("=" * 74)
    for n in NS:
        pl = results[f"plain   n={n}"]["win"]["MAX"]
        print(f"\n  n={n} (plain MAX Sharpe {pl[1]:.2f}, DD {pl[2]*100:.0f}%):")
        for cap in CAPS:
            r = results[f"cap{cap} n={n}"]["win"]["MAX"]
            print(f"    cap={cap}: dSharpe {r[1]-pl[1]:+.2f}  dDD {(r[2]-pl[2])*100:+.0f}pp  "
                  f"dCAGR {(r[0]-pl[0])*100:+.1f}pp")

    return results


if __name__ == "__main__":
    main()
