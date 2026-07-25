"""Fetch + convert the free S&P 500 historical membership dataset.

Source: github.com/fja05680/sp500 (MIT) -- "S&P 500 Historical Components &
Changes (Updated).csv", ~2,700 date snapshots from 1996 to the present, wide
format (date, comma-joined tickers). Converts to the long (date,ticker) format
pit_universe.CSVAdapter expects and writes data/sp500_membership.csv.

This is TRUE point-in-time S&P 500 membership -- the research universe for
survivorship_stress.py / edge_ic.py (run with
PIT_MEMBERSHIP_CSV=data/sp500_membership.csv). It is deliberately NOT written
to data/russell1000_membership.csv: that path auto-activates the PIT universe
for the whole product, and the product trades the top-1000 proxy.

Run:  .venv-mac/bin/python fetch_membership.py
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
       "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
OUT = Path(__file__).resolve().parent / "data" / "sp500_membership.csv"


def main():
    print(f"Downloading {URL} ...")
    raw = urllib.request.urlopen(URL, timeout=120).read()
    df = pd.read_csv(io.BytesIO(raw))
    rows = []
    for _, r in df.iterrows():
        for t in str(r["tickers"]).split(","):
            t = t.strip()
            if t:
                rows.append((r["date"], t))
    out = pd.DataFrame(rows, columns=["date", "ticker"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out):,} rows, {out['date'].nunique()} snapshots, "
          f"{out['date'].min()} -> {out['date'].max()}")


if __name__ == "__main__":
    main()
