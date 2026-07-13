"""The Edge -- Stage 2: does a trend-QUALITY tilt improve the net book? (GPT 5.2)

IC screen (edge_test_accel.py) found the accel-geometry curvature idea fails, but
trend efficiency (Kaufman ER = net move / total path) and R^2 trend quality show a
clean +0.95 monotonic staircase and POSITIVE incremental IC over accel -- the best
new lead of the campaign, though sub-t3 as a standalone. Since our priority is
risk-adjusted shape, test it the only way that matters: as a TILT on top of accel
in the full net portfolio, vs the pure-accel baseline.

Blended score = z(accel) + w * z(trend_eff)  (and an r2 variant), full spec
(floor + corr-cap + regime + 10bps + growth-mix 75), hold=42, n in {7,10}.
Graduates only if it improves the net book -- ideally Sharpe/maxDD -- not just CAGR.

Reuses edge_test_skip's net evaluator + edge_test_accel's signal builder.
Does NOT modify edge_lib. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_test_skip as SK
from edge_test_skip import run_spec, window_row
from edge_signal_ic import _raw_prep
from edge_test_accel import sig_at


def attach(pan, prep):
    """Attach trend_eff / r2_126 / trend252 to pan.panels IN PLACE (test-only)."""
    for i, df in enumerate(pan.panels):
        if "trend_eff" in df.columns:
            continue
        dt = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        recs = {}
        for ck in df["company_key"]:
            P = prep.get(ck)
            if P is None:
                continue
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < 252 or pos >= len(P["arr"]):
                continue
            s = sig_at(P, pos)
            recs[ck] = {"trend_eff": s["trend_eff"], "r2_126": s["r2_126"],
                        "trend252": s["trend252"]}
        add = pd.DataFrame.from_dict(recs, orient="index")
        pan.panels[i] = df.set_index("company_key").join(add).reset_index()
        if (i + 1) % 60 == 0:
            print(f"    attach {i+1}/{pan.T}")


def main():
    pan = E.load_edge_panel(hold=42)
    print("Attaching trend-quality columns ...")
    attach(pan, _raw_prep())
    SK.GROWTH_MIX = 0.75

    print(f"\n{'='*104}\nStage 2 — accel + trend-quality tilt, full spec net, hold=42"
          f"\n{'='*104}")
    for n in (7, 10):
        SK.NBASKET = n
        print(f"\nn={n}:  window: CAGR/Sharpe/maxDD (excess vs S&P)   [baseline = pure accel]")
        specs = [
            ("accel (baseline)",            {"accel": 1.0}),
            ("accel + 0.25*trend_eff",      {"accel": 1.0, "trend_eff": 0.25}),
            ("accel + 0.50*trend_eff",      {"accel": 1.0, "trend_eff": 0.50}),
            ("accel + 1.0*trend_eff",       {"accel": 1.0, "trend_eff": 1.0}),
            ("accel + 0.50*r2_126",         {"accel": 1.0, "r2_126": 0.50}),
            ("accel + 0.25te + 0.25r2",     {"accel": 1.0, "trend_eff": 0.25, "r2_126": 0.25}),
        ]
        for label, w in specs:
            net, avg_to = run_spec(pan, w)
            print(window_row(pan, net, label, avg_to))
    print("\nDONE.  Adopt only if the tilt lifts Sharpe/maxDD without giving back CAGR.")


if __name__ == "__main__":
    main()
