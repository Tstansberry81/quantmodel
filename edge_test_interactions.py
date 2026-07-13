"""The Edge -- regularized INTERACTIONS (GPT audit 5.4), walk-forward.

Our earlier elastic-net (edge_test_metamodel.py) used only LINEAR features and found
no structure. GPT's actual suggestion is interaction terms -- e.g. acceleration is
valuable only when the trend is already up and volatility is moderate:
    E[R] = b0 + b1*MOM + b2*ACCEL + b3*(MOM x ACCEL) + b4*(ACCEL x VOL) + ...
This tests exactly that: base signals PLUS pairwise interactions, walk-forward
ElasticNetCV (expanding window, strictly-prior training), vs the accel baseline.

Reuses the metamodel harness (walk_forward / book_from_scores / net_from_holds)
with an extended feature builder. Graduates only if OOS IC clears the bar AND it
beats accel in the net book. hold=42. Does NOT modify edge_lib. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_test_metamodel as MM
from edge_signal_ic import spearman, t_stat

BASE = ["accel", "ret_63", "ret_21", "ret_126", "hi_252", "vol_21", "rev_growth"]
# pairwise interactions GPT motivates: momentum x accel, accel x vol, trend x accel...
INTER = [("accel", "ret_63"), ("accel", "vol_21"), ("accel", "hi_252"),
         ("ret_63", "vol_21"), ("accel", "rev_growth"), ("ret_63", "hi_252")]
FEATS = BASE + [f"{a}*{b}" for a, b in INTER]
FLOOR = 2e9


def prep_interactions(pan):
    """Per-period standardized features (base + interactions) + demeaned target."""
    periods = []
    for df in pan.panels:
        d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= FLOOR].dropna(subset=["fwd_ret"]).copy()
        if len(d) < 40:
            periods.append(None); continue
        # z-score base columns first (so interactions are products of standardized signals)
        Z = {}
        for c in BASE:
            v = pd.to_numeric(d[c], errors="coerce")
            mu, sd = v.mean(), v.std()
            Z[c] = ((v - mu) / sd).fillna(0.0).to_numpy() if sd and sd > 0 else np.zeros(len(d))
        cols = [Z[c] for c in BASE]
        for a, b in INTER:
            cols.append(Z[a] * Z[b])
        X = np.column_stack(cols)
        y = pd.to_numeric(d["fwd_ret"], errors="coerce"); y = (y - y.mean()).to_numpy()
        periods.append({"cks": d["company_key"].to_numpy(), "X": X, "y": y,
                        "accel": pd.to_numeric(d["accel"], errors="coerce").to_numpy(),
                        "fwd": pd.to_numeric(d["fwd_ret"], errors="coerce").to_numpy()})
    return periods


def main():
    hold = 42
    pan = E.load_edge_panel(hold=hold); ppy = pan.ppy
    print(f"hold={hold} — walk-forward elastic net WITH interactions ({len(FEATS)} features)")
    periods = prep_interactions(pan)
    preds, coefs = MM.walk_forward(periods)

    ic_ml, ic_ac = [], []
    for i, p in enumerate(periods):
        if p is None or preds[i] is None:
            continue
        ic_ml.append(spearman(preds[i], p["fwd"])); ic_ac.append(spearman(p["accel"], p["fwd"]))
    ic_ml, ic_ac = np.array(ic_ml), np.array(ic_ac)
    oos = [i for i, p in enumerate(periods) if p is not None and preds[i] is not None]
    print(f"\nOOS window: {len(oos)} rebalances {pan.bdates[oos[0]].date()} -> {pan.bdates[oos[-1]].date()}")
    print(f"   elastic-net + interactions OOS IC: {np.nanmean(ic_ml)*100:+5.2f}%  t={t_stat(ic_ml):+5.2f}")
    print(f"   accel                       OOS IC: {np.nanmean(ic_ac)*100:+5.2f}%  t={t_stat(ic_ac):+5.2f}")

    print("\nAvg coefficients (|top 8|):")
    order = np.argsort(-np.abs(coefs.mean(0)))[:8]
    for j in order:
        print(f"   {FEATS[j]:<16} {coefs.mean(0)[j]:+.4f}  (nonzero {100*np.mean(coefs[:,j]!=0):.0f}%)")

    spx = pan.spxf[oos]
    print(f"\nNET portfolios over OOS (period returns, regime+10bps) vs accel baseline:")
    for n in (10, 7):
        acc = []
        for i in oos:
            dd = pan.panels[i][pan.panels[i]["pit_mcap"] >= FLOOR]
            acc.append(E.corr_cap_select(dd, asof=pan.bdates[i], n=n, weights={"accel": 1.0},
                                         cap=0.5, lookback=126))
        r_ac, to_ac = MM.net_from_holds(pan, acc, oos)
        h_ml, i_ml = MM.book_from_scores(periods, preds, n, meta=False)
        r_ml, _ = MM.net_from_holds(pan, h_ml, i_ml)
        h_me, i_me = MM.book_from_scores(periods, preds, n, meta=True)
        r_me, _ = MM.net_from_holds(pan, h_me, i_me)
        print(f"  n={n}:")
        MM.perf_row(f"accel baseline", r_ac, ppy, spx)
        MM.perf_row(f"EN+interactions full X-sec", r_ml, ppy, spx)
        MM.perf_row(f"EN+interactions META on accel-top40", r_me, ppy, spx)
    print("\nDONE.  Interactions graduate only if they beat accel net + OOS IC t>~3.")


if __name__ == "__main__":
    main()
