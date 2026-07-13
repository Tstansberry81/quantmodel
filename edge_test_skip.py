"""The Edge -- skip-period momentum + reversal-filter experiments.

Motivated by edge_signal_ic.py's finding: the one significant price-only effect
is SHORT-TERM REVERSAL (ret_21 IC -2.2%/t-2.4 @1mo). The Edge ranks on accel
using prices right up to the rebalance date, so it partly buys names at a
short-term-overbought peak and eats the reversal on entry.

Two remedies tested here (both literature-standard, pre-registered, low-dof):
  A. SKIP-PERIOD signals -- compute the formation window ending s days ago
     (s in {5,10,21}), the classic "12-1 skip-month" idea applied to accel,
     plus canonical 12-1 momentum itself (never tested on the Edge before).
  B. REVERSAL ENTRY FILTER -- rank on accel as today, but refuse to buy names
     in the top decile/quintile of trailing 1-month return (overbought), or
     blend a small negative ret_21 weight.

Stage 1: rank-IC scorecard (all ~795 names x all rebalances) at hold=21 and 42.
Stage 2: full-spec net-of-cost portfolio test at hold=42 (the product clock):
         $2B floor + corr-cap 0.5 + regime 25% + 10bps + growth-mix 0.75.

Does NOT modify edge_lib.py / qmodel/. Nothing is committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import _raw_prep, scorecard, spearman

LB = E.LB

# spec constants (mirror EDGE_SPEC; kept local so nothing central changes)
MCAP_FLOOR, CORR_CAP, CORR_LB = 2e9, 0.50, 126
REGIME_EXPO, COST_BPS = 0.25, 10.0
GROWTH_MIX, GROWTH_THRESH, NBASKET = 0.75, 0.15, 10


# ---- skip-period signals ------------------------------------------------------
def _skip_signals_at(P, pos):
    """Skip-window variants at trailing position pos (data through pos only)."""
    arr = P["arr"]
    out = {}
    for s in (5, 10, 21):
        p = pos - s
        out[f"accel_s{s}"] = (arr[p] / arr[p - 63] - 1) - (arr[p - 63] / arr[p - 126] - 1)
        out[f"ret63_s{s}"] = arr[p] / arr[p - 63] - 1
    # canonical 12-1 momentum: 12-month return excluding the most recent month
    out["mom_12_1"] = arr[pos - 21] / arr[pos - 251] - 1
    # 6-1: six-month return excluding the last month
    out["ret126_s21"] = arr[pos - 21] / arr[pos - 147] - 1
    return out


def attach_skip_signals(pan):
    """Add skip-variant columns IN PLACE on pan.panels (test-only, in-process)."""
    prep = _raw_prep()
    for i, df in enumerate(pan.panels):
        dt = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        recs = {}
        for ck in df["company_key"]:
            P = prep.get(ck)
            if P is None:
                continue
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB or pos >= len(P["arr"]):
                continue
            recs[ck] = _skip_signals_at(P, pos)
        add = pd.DataFrame.from_dict(recs, orient="index")
        pan.panels[i] = df.set_index("company_key").join(add).reset_index()
        if (i + 1) % 40 == 0:
            print(f"    skip signals {i+1}/{len(pan.panels)}")


# ---- stage 2: full-spec net portfolio -----------------------------------------
def run_spec(pan, weights, rev_filter_q=None, label=""):
    """Full Edge spec (floor + corr cap + regime + costs + growth mix) with an
    arbitrary signal weighting and an optional reversal entry filter that drops
    names above the rev_filter_q quantile of trailing 1-month return."""
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= MCAP_FLOOR]
        if rev_filter_q is not None:
            r21 = pd.to_numeric(d["ret_21"], errors="coerce")
            d = d[r21 <= r21.quantile(rev_filter_q)]
        cks = E._blend_select(d, pan.bdates[i], NBASKET, weights,
                              CORR_CAP, CORR_LB, GROWTH_MIX, GROWTH_THRESH)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    gross = E.period_returns(pan, holds)
    gross = np.where(pan.ma200_on, gross, REGIME_EXPO * gross + (1 - REGIME_EXPO) * pan.rf_per)
    net = gross - (COST_BPS / 1e4) * np.array(turn)
    return net, float(np.mean(turn))


def window_row(pan, net, label, avg_to):
    ppy = pan.ppy
    parts = [f"{label:<28}"]
    for wname, yrs in (("1Y", 1), ("2Y", 2), ("5Y", 5), ("MAX", None)):
        kk = len(net) if yrs is None else min(int(round(yrs * ppy)), len(net))
        c, dd, sh = E.perf(net[-kk:], ppy)
        cs, _, _ = E.perf(pan.spxf[-kk:], ppy)
        parts.append(f"{wname}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}% ({(c-cs)*100:+5.1f})")
    parts.append(f"to={avg_to:.2f}")
    return "  ".join(parts)


def main():
    # ---------- Stage 1: IC at both clocks ----------
    skip_cols = ["accel_s5", "accel_s10", "accel_s21", "ret63_s5", "ret63_s10",
                 "ret63_s21", "mom_12_1", "ret126_s21"]
    for hold in (21, 42):
        print(f"\n{'='*70}\nSTAGE 1 -- rank-IC of skip variants at hold={hold}\n{'='*70}")
        pan = E.load_edge_panel(hold=hold)
        print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}")
        attach_skip_signals(pan)
        scorecard(pan.panels, pan.bdates, ["accel", "ret_63"] + skip_cols)

    # ---------- Stage 2: full-spec net portfolio at the product clock ----------
    print(f"\n{'='*70}\nSTAGE 2 -- full-spec NET portfolio at hold=42 "
          f"(floor+corrcap+regime+10bps+mix75)\n{'='*70}")
    pan = E.load_edge_panel(hold=42)   # already has skip columns attached (same object)

    specs = [
        ("accel (BASELINE)",        {"accel": 1.0},     None),
        ("accel_s5",                {"accel_s5": 1.0},  None),
        ("accel_s10",               {"accel_s10": 1.0}, None),
        ("accel_s21",               {"accel_s21": 1.0}, None),
        ("mom_12_1",                {"mom_12_1": 1.0},  None),
        ("ret126_s21 (6-1)",        {"ret126_s21": 1.0}, None),
        ("accel + drop top20% r21", {"accel": 1.0},     0.80),
        ("accel + drop top10% r21", {"accel": 1.0},     0.90),
        ("accel - 0.25*ret_21",     {"accel": 1.0, "ret_21": -0.25}, None),
        ("accel_s10 + drop top20%", {"accel_s10": 1.0}, 0.80),
    ]
    print(f"{'spec':<28}  window: CAGR/Sharpe/maxDD (excess vs S&P)")
    for label, w, q in specs:
        net, avg_to = run_spec(pan, w, rev_filter_q=q)
        print(window_row(pan, net, label, avg_to))
    print("\nDONE. (to = avg per-rebalance turnover; costs already netted at 10bps)")


if __name__ == "__main__":
    main()
