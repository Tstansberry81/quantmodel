"""The Edge -- Post-Earnings-Announcement Drift (PEAD) test, round-5 earnings dynamics.

Time-series SUE (Standardized Unexpected Earnings, seasonal-random-walk form of
Foster-Olsen-Shevlin 1984 / Bernard-Thomas 1989) computed from REPORTED quarterly
diluted EPS + announcement dates only -- NO analyst estimates needed:

    dEPS_q = EPS_q - EPS_{q-4}                        (seasonal difference)
    SUE_q  = dEPS_q / std(dEPS over trailing 8 q)     (standardized)

PEAD = stocks with high SUE drift UP over the following ~1 quarter. Tested as a
cross-sectional signal on the panel, PIT-safe (uses only the most recent earnings
announcement on/before each rebalance). We report the plain SUE and a
"recent-announcement" SUE (only names that announced within the last ~63 trading
days -- the drift window), plus incremental IC vs accel (does it ADD?).

Data: data/cache/quarterly_eps.pkl (fiscal_quarterly_eps.py). ~2016-2026 window.
Usage: .venv/Scripts/python.exe edge_test_pead.py [--hold 42]
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import argparse, pickle
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import scorecard

COLS = ["sue", "sue_recent", "sue_pos", "days_since"]
DRIFT_DAYS = 63          # ~1 quarter: the PEAD drift window


def _sue(df):
    """SUE series (index = announcement date) from a name's quarterly EPS."""
    eps = pd.to_numeric(df["eps"], errors="coerce")
    d = eps - eps.shift(4)                                  # seasonal (YoY quarter) diff
    sd = d.rolling(8, min_periods=4).std()
    return d / sd


def _prep(qeps):
    prep = {}
    for ck, df in qeps.items():
        s = _sue(df).dropna()
        if len(s) < 2:
            continue
        prep[ck] = (np.asarray(s.index.values, "datetime64[ns]"),
                    np.asarray(s.values, float))
    return prep


def attach(pan, qeps):
    prep = _prep(qeps)
    cover = {}; panels = []
    for i, pdf in enumerate(pan.panels):
        d = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        recs = {}
        for ck in pdf["company_key"]:
            pr = prep.get(ck)
            if pr is None:
                continue
            annidx, sarr = pr
            pos = int(np.searchsorted(annidx, d, side="right")) - 1   # last announced <= d
            if pos < 0:
                continue
            sue = float(sarr[pos])
            dsi = (d - annidx[pos]) / np.timedelta64(1, "D")
            recs[ck] = {"sue": sue,
                        "sue_recent": sue if dsi <= DRIFT_DAYS * 7 / 5 else np.nan,  # ~63 trading d
                        "sue_pos": 1.0 if sue > 0 else 0.0,
                        "days_since": float(dsi)}
        add = pd.DataFrame.from_dict(recs, orient="index").reindex(columns=COLS)
        panels.append(pdf.set_index("company_key").join(add).reset_index())
        for c in COLS:
            cover.setdefault(c, [0, 0])
            cover[c][0] += int(panels[-1][c].notna().sum()); cover[c][1] += len(panels[-1])
    return panels, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=42)
    ap.add_argument("--pkl", default="data/cache/quarterly_eps.pkl")
    args = ap.parse_args()

    qeps = pickle.load(open(args.pkl, "rb"))
    print(f"Loaded quarterly EPS for {len(qeps)} names")
    pan = E.load_edge_panel(hold=args.hold)
    # restrict the panel window to where SUE coverage is real (~2016+)
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}")
    panels, cover = attach(pan, qeps)

    print("\nCOVERAGE (share of name-rebalance rows with a value):")
    for c in COLS:
        n, d = cover[c]; print(f"    {c:<12} {n/d*100:5.1f}%  ({n:,}/{d:,})")
    # how many rows are within the drift window (recent announcement)
    print(f"\nExpected signs: sue +, sue_recent + (POS IC = PEAD works). Test window ~2016-2026.")
    print("\nPEAD SCORECARD (rank-IC vs fwd ret; incr = after accel):")
    scorecard(panels, pan.bdates, ["sue", "sue_recent", "sue_pos"])


if __name__ == "__main__":
    main()
