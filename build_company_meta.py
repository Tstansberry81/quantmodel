"""Build data/artifacts/company_meta.json — the industry / location sidecar.

WHY A SIDECAR AND NOT THE INGEST
--------------------------------
Sharadar's TICKERS table carries `industry` (e.g. "Semiconductors"), which is a
lot more useful on a stock list than the 11 GICS-style `sector` buckets the
artifact already stores. But sharadar_ingest.py does not keep it, and re-running
the ingest to add one string means re-downloading and re-pickling ~2.5GB of
prices for a 200KB fact.

So this reads the TICKERS export directly and writes a small JSON next to the
artifact. make_data_bundle.py globs data/artifacts/*, so it ships automatically.

ONE FACT, ONE SOURCE: this file deliberately does NOT carry `sector`. The
artifact already has sector, and a second copy of the same field in a file that
is refreshed on a different schedule is how two parts of a page end up
disagreeing about the same stock. Industry and location live here; sector lives
in the artifact; neither is duplicated.

Keyed by TICKER, not company_key. The artifact's company_key embeds the exchange
(NASDAQ_NVDA), and a name that changes listing venue would silently lose its
description. Tickers can be reused across decades, but this is display text for
names in a CURRENT book, so the reuse case cannot arise.

    python build_company_meta.py            # writes data/artifacts/company_meta.json
    python build_company_meta.py --all      # every equity ticker, not just artifact names
"""
from __future__ import annotations
import argparse
import json
import pickle
import zipfile

import pandas as pd

import config

KEEP = ("name", "industry", "sicindustry", "category", "location", "companysite")
OUT = config.ARTIFACT_DIR / "company_meta.json"
TICKERS_ZIP = config.ROOT / "data" / "sharadar" / "TICKERS.zip"


def _artifact_tickers() -> set[str] | None:
    """Tickers present in the current artifact, so the sidecar stays small.
    None if the artifact isn't readable (then we keep everything)."""
    art = config.ARTIFACT_DIR / "backtest_data.pkl"
    if not art.exists():
        return None
    with open(art, "rb") as fh:
        data = pickle.load(fh)
    return {str(b.get("meta", {}).get("ticker") or "").upper()
            for b in data.values()} - {""}


def build(limit_to_artifact: bool = True) -> dict:
    if not TICKERS_ZIP.exists():
        raise SystemExit(
            f"build_company_meta: {TICKERS_ZIP} not found. This needs the Sharadar\n"
            f"  TICKERS bulk export, which only exists on the research machine --\n"
            f"  the deploy host reads the JSON this script produces, out of the bundle.")
    with zipfile.ZipFile(TICKERS_ZIP) as z:
        df = pd.read_csv(z.open(z.namelist()[0]),
                         usecols=["table", "ticker", *KEEP], low_memory=False)
    df = df[df["table"] == "SEP"].drop(columns=["table"])
    # last row wins: same rule sharadar_ingest uses, so the two files agree on
    # which row is authoritative for a re-symboled ticker.
    df = df.drop_duplicates("ticker", keep="last")
    df["ticker"] = df["ticker"].astype(str).str.upper()

    keep = _artifact_tickers() if limit_to_artifact else None
    if keep:
        df = df[df["ticker"].isin(keep)]

    out = {}
    for row in df.itertuples(index=False):
        rec = {k: ("" if pd.isna(getattr(row, k, None)) else str(getattr(row, k)).strip())
               for k in KEEP}
        if any(rec.values()):
            out[row.ticker] = {k: v for k, v in rec.items() if v}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="keep every equity ticker, not just the ones in the artifact")
    args = ap.parse_args()

    meta = build(limit_to_artifact=not args.all)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, indent=0, sort_keys=True), encoding="utf-8")
    n_ind = sum(1 for v in meta.values() if v.get("industry"))
    print(f"wrote {OUT} — {len(meta):,} tickers, {n_ind:,} with an industry "
          f"({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
