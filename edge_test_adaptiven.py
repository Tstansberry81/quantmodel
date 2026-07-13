"""The Edge -- adaptive 5-10 basket by a deterministic breadth rule (GPT audit 5.6).

Instead of a fixed n, size the book K in [5,10] from the cross-section's conviction,
using only time-t data (no lookahead). Two rules:
  * threshold: K = clip(5, 10, #names with z(accel) > c)   -- breadth of strong names
  * gap:       cut at the largest score gap among ranks 5..10 -- natural break

Evaluated under the HONEST product framing (staggered sleeves + daily buy-and-hold),
vs fixed n=7 and n=10. Graduates only if adaptive sizing improves risk-adjusted
shape (Sharpe / maxDD) -- not just CAGR. Reuses edge_lib's daily sim. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as D

S = E.EDGE_SPEC


def adaptive_K(d, rule):
    """Deterministic K in [5,10] from the floored candidates' accel z-scores."""
    z = E._z(pd.to_numeric(d["accel"], errors="coerce")).to_numpy()
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return max(5, min(10, len(z)))
    if rule.startswith("thr"):
        c = float(rule.split("_")[1])
        k = int((z > c).sum())
    else:  # gap: biggest drop among ranks 5..10
        s = np.sort(z)[::-1]
        gaps = s[4:10] - s[5:11]           # gap after rank 5,6,...,10
        k = 5 + int(np.argmax(gaps))
    return int(np.clip(k, 5, 10))


def holds_for(pan, rule):
    """Variable-length books per rebalance under a K rule (or fixed int)."""
    holds, turn, prev, ks = [], [], None, []
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= S["mcap_floor"]]
        k = rule if isinstance(rule, int) else adaptive_K(d, rule)
        ks.append(k)
        cks = E._blend_select(d, pan.bdates[i], k, dict(S["signal"]), S["corr_cap"],
                              S["corr_lookback"], S["growth_mix"], S["growth_thresh"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn), np.array(ks)


def staggered_daily(M, rule):
    """Combined 50/50 daily series for a rule across the two product sleeves."""
    series, kmeans = [], []
    for off in (0, 21):
        pan = E.load_edge_panel(hold=S["hold"], offset_days=off)
        h, t, ks = holds_for(pan, rule)
        series.append(E._sleeve_daily(pan, h, t, M, S["hold"], S["regime_expo"], S["cost_bps"]))
        kmeans.append(ks.mean())
    d = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
    return d, float(np.mean(kmeans))


def perf_windows(r, spx, label, kmean=None):
    r = np.asarray(r, float); spx = np.asarray(spx, float); T = len(r)
    parts = [f"  {label:<22}"]
    for wn, k in (("1Y", 252), ("2Y", 504), ("5Y", 1260), ("MAX", None)):
        seg = r if k is None else r[-min(k, T):]; ss = spx if k is None else spx[-min(k, T):]
        c, dd, sh = E._perf_daily(seg); cs, _, _ = E._perf_daily(ss)
        parts.append(f"{wn}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}% ({(c-cs)*100:+5.1f})")
    if kmean is not None:
        parts.append(f"avgK {kmean:.1f}")
    print("   ".join(parts))


def main():
    M = D.daily_return_matrix()
    bm = D.benchmarks(); spx_full = bm["SP500"]
    print("Adaptive 5-10 basket vs fixed, staggered + daily (honest product framing)\n" + "=" * 100)
    rules = [(10, "fixed n=10"), (7, "fixed n=7"),
             ("thr_1.0", "adaptive thr>1.0"), ("thr_1.5", "adaptive thr>1.5"),
             ("gap", "adaptive gap 5-10")]
    for rule, label in rules:
        d, kmean = staggered_daily(M, rule)
        spx = spx_full.reindex(d.index, method="ffill").pct_change().fillna(0.0).to_numpy()
        perf_windows(d.to_numpy(), spx, label, None if isinstance(rule, int) else kmean)
    print("\nDONE.  Adopt only if an adaptive rule lifts Sharpe / shrinks maxDD vs fixed.")


if __name__ == "__main__":
    main()
