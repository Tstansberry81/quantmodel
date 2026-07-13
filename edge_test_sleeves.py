"""The Edge -- staggered sleeves (GPT audit 5.7) + honest daily buy-and-hold sim.

Not a cross-sectional signal (those are tapped out on this universe) but a
PORTFOLIO-CONSTRUCTION change: run two 2-month sleeves offset by ~1 month, each
held 42 trading days, combined 50/50. Attacks a real fragility -- results depend
on the arbitrary rebalance calendar -- and should smooth returns / reduce
date-luck / lower effective turnover even if it doesn't add raw return.

Implemented as a DAILY buy-and-hold simulation (weights drift within each holding
window; regime scales the whole window; cost charged on the entry day) -- this
also addresses the audit's 2.2 point (simulate drifting weights, not daily
rebalancing). Compares, all daily-simulated the same way:
  A. single sleeve (offset 0)      -- the current cadence
  B. single sleeve (offset 21d)    -- same strategy, different calendar (fragility check)
  C. combined 0.5*A + 0.5*B        -- the staggered book

Uses the new load_edge_panel(offset_days=) hook (offset 0 == current product).
Graduates if C improves risk-adjusted shape (Sharpe / maxDD / smoothness) at
comparable CAGR. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config
import edge_lib as E
import edge_data as D

S = E.EDGE_SPEC
RF_D = config.RISK_FREE_ANNUAL / 252.0


def sleeve_holds(pan, n):
    """Full-spec selection per rebalance + membership turnover (mirrors _edge_full)."""
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= S["mcap_floor"]]
        cks = E._blend_select(d, pan.bdates[i], n, dict(S["signal"]), S["corr_cap"],
                              S["corr_lookback"], S["growth_mix"], S["growth_thresh"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, turn


def sleeve_daily(pan, holds, turn, M, hold, expo, cost_bps):
    """Daily return series: buy-and-hold each book for `hold` days from t+1, weights
    drifting, regime scaling the window, cost on the entry day."""
    cal = M.index
    out = pd.Series(np.nan, index=cal)
    for i, cks in enumerate(holds):
        loc = int(cal.searchsorted(pan.bdates[i], side="right"))     # t+1 entry (exec lag)
        win = cal[loc: loc + hold]
        names = [c for c in cks if c in M.columns]
        if len(win) == 0 or not names:
            continue
        sub = np.nan_to_num(M.loc[win, names].to_numpy(float))
        w = np.full(len(names), 1.0 / len(names))
        on = bool(pan.ma200_on[i])
        rets = np.empty(len(win))
        for dd in range(len(win)):
            r = sub[dd]
            pr = float(np.dot(w, r))
            if not on:
                pr = expo * pr + (1 - expo) * RF_D
            rets[dd] = pr
            g = w * (1 + r); ssum = g.sum()
            if ssum > 0:
                w = g / ssum
        rets[0] -= (cost_bps / 1e4) * turn[i]                        # entry-day cost
        out.loc[win] = rets
    return out


def perf_daily(r, spx_d):
    r = np.asarray(r, float)
    windows = (("1Y", 252), ("2Y", 504), ("5Y", 1260), ("MAX", None))
    parts = []
    for wn, k in windows:
        seg = r if k is None else r[-k:]
        ss = spx_d if k is None else spx_d[-k:]
        c, dd, sh = E.perf(seg, 252); cs, _, _ = E.perf(ss, 252)
        parts.append(f"{wn}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}% ({(c-cs)*100:+5.1f})")
    return "   ".join(parts)


def main():
    M = D.daily_return_matrix()
    bm = D.benchmarks(); spx = bm["SP500"].dropna()
    hold = S["hold"]
    print(f"Staggered sleeves — hold={hold}, offset=21d (half a period); daily buy-and-hold sim\n")
    for n in (10, 7):
        panA = E.load_edge_panel(hold=hold, offset_days=0)
        panB = E.load_edge_panel(hold=hold, offset_days=21)
        hA, tA = sleeve_holds(panA, n)
        hB, tB = sleeve_holds(panB, n)
        dA = sleeve_daily(panA, hA, tA, M, hold, S["regime_expo"], S["cost_bps"])
        dB = sleeve_daily(panB, hB, tB, M, hold, S["regime_expo"], S["cost_bps"])
        # align on the overlap where both sleeves are active
        both = pd.concat([dA, dB], axis=1).dropna()
        a, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
        comb = 0.5 * a + 0.5 * b
        spx_d = spx.reindex(both.index, method="ffill").pct_change().fillna(0.0).to_numpy()
        # smoothness: annualized vol of the daily stream (lower = smoother)
        volA = np.std(a) * np.sqrt(252); volC = np.std(comb) * np.sqrt(252)
        print(f"=== n={n} (daily buy-and-hold, {len(both)} days) — CAGR/Sharpe/maxDD (excess vs S&P) ===")
        print(f"  A. single sleeve (offset 0)   {perf_daily(a, spx_d)}   vol {volA*100:.0f}%")
        print(f"  B. single sleeve (offset 21)  {perf_daily(b, spx_d)}")
        print(f"  C. staggered 0.5A+0.5B        {perf_daily(comb, spx_d)}   vol {volC*100:.0f}%")
        print(f"     turnover/rebalance: A {np.mean(tA):.2f}  B {np.mean(tB):.2f}  "
              f"(combined trades half the book each month)\n")
    print("DONE.  Adopt if C lifts Sharpe / shrinks maxDD or vol at comparable CAGR.")


if __name__ == "__main__":
    main()
