"""Pull the Sharadar bulk tables and build the Edge's backtest artifact.

WHY THIS EXISTS
---------------
The fiscal.ai artifact carries 1,111 names of which 111 are delisted -- and every
one of those died in 2024 or later. There is effectively NO delisted history
before 2024, so the backtest never holds a stock through its death and the
survivorship question could only be bounded, never answered. Sharadar's SEP
covers 21,946 tickers of which 15,634 are delisted, with prices back to 1998.

This module downloads the bulk exports once and rebuilds
`data/artifacts/backtest_data.pkl` in EXACTLY the shape edge_data.load_bt_data()
already returns:

    {company_key: {"prices": Series, "fund_hist": DataFrame, "meta": {...}}}

so edge_lib, the tracker, the export and every research script keep working
untouched -- and fiscal.ai vs Sharadar becomes a clean A/B on data quality alone.

POINT-IN-TIME DISCIPLINE (see sharadar_kit/CLAUDE.md)
  * SF1 dimension ART only (as-reported TTM). MR* are restated -> look-ahead.
  * Fundamentals are indexed by `datekey` (SEC filing date) + a filing-lag
    censor, never by `calendardate` (period end, unobservable until filed).
  * Prices are `closeadj` (split- AND dividend-adjusted) for total return.
  * Delisted tickers are kept. Dead names are re-symboled at bankruptcy
    (LEH -> LEHMQ, Bear Stearns -> BSC1), so identity is `permaticker`.

Run:
    .venv-mac/bin/python sharadar_ingest.py download     # ~1.7GB, once
    .venv-mac/bin/python sharadar_ingest.py build        # -> artifacts
    .venv-mac/bin/python sharadar_ingest.py all
"""
from __future__ import annotations

import io
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config                      # importing config loads .env
import sharadar_client as S

RAW = config.DATA_DIR / "sharadar"
RAW.mkdir(parents=True, exist_ok=True)

# Bulk tables we need. SFP (benchmarks) is pulled per-ticker via the API
# instead -- we only want a handful of ETFs, not the whole 300MB fund table.
BULK_TABLES = ("TICKERS", "SF1", "SEP")

# The filing-lag censor, in trading days, applied on top of `datekey`.
# The kit's house rule is >= 1; do not lower it.
FILING_LAG_DAYS = 1

# Columns we keep from SF1/ART. `marketcap` drives the point-in-time universe
# ranking; `revenueusd` drives the growth mix.
SF1_COLS = ["ticker", "datekey", "calendardate", "marketcap", "revenueusd",
            # profitability / quality
            "roic", "roe", "roa", "grossmargin", "netmargin",
            # cash generation
            "fcf", "ncfo", "capex", "netinc", "ebitda",
            # leverage / solvency (assets also scales GP/assets and accruals)
            "de", "debt", "equity", "currentratio", "assets",
            # valuation as filed (stale between filings -- DAILY is preferred,
            # these are the fallback when the DAILY table isn't downloaded)
            "pe", "pb", "ps", "evebitda",
            # inputs for the academically-strongest quality/anomaly measures:
            # Novy-Marx gross profitability, Sloan accruals, net share issuance
            "sharesbas", "shareswa"]

# Daily point-in-time valuation. Sharadar recomputes these EVERY trading day
# off the then-known fundamentals and that day's price, so a name whose price
# halves shows the new multiple immediately. SF1's own pe/pb/ps are as-of the
# filing and stay stale for up to a quarter -- fatal for a valuation screen,
# which is exactly the thing being tested here.
DAILY_COLS = ["ticker", "date", "marketcap", "pe", "pb", "ps", "evebitda", "ev"]

# Fundamental columns attached to every panel row (on top of market cap and
# revenue growth, which the model already used).
FUND_ATTACH = ("roic", "roe", "roa", "grossmargin", "netmargin", "fcf_margin",
               "ebitda_margin", "growth_fcf_1y", "growth_netinc_1y",
               "de", "debt_ebitda", "currentratio",
               "pe", "pb", "ps", "evebitda",
               "gp_assets", "accruals", "share_issuance", "fcf")

# Price-history cutoff: keep any name that EVER reached this market cap. Set
# well below the product's $2B liquidity floor so no selectable name is lost,
# while keeping the SEP scan tractable.
MCAP_KEEP = 1e9


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def _export_link(sc: S.SharadarClient, table: str) -> str:
    """Ask for the bulk export and return the S3 link once it's fresh."""
    url = f"{S.BASE_URL}/SHARADAR/{table}.csv"
    for attempt in range(30):
        r = sc._session.get(url, params={"api_key": sc.api_key,
                                         "qopts.export": "true"}, timeout=90)
        if r.status_code != 200:
            raise S.SharadarError(f"{table} export failed {r.status_code}: {r.text[:200]}")
        last = r.text.strip().splitlines()[-1]
        parts = last.split(",")
        link, status = parts[0], (parts[1] if len(parts) > 1 else "")
        if link.startswith("http") and status.strip() == "fresh":
            return link
        print(f"  {table}: export {status or 'regenerating'} — waiting…", flush=True)
        time.sleep(10)
    raise S.SharadarError(f"{table}: export never became fresh")


def download(tables=BULK_TABLES, force: bool = False) -> None:
    sc = S.SharadarClient()
    for t in tables:
        out = RAW / f"{t}.zip"
        if out.exists() and not force:
            print(f"[skip] {t}: already have {out.name} ({out.stat().st_size/1e6:.0f} MB)")
            continue
        link = _export_link(sc, t)
        print(f"[get ] {t} …", flush=True)
        with sc._session.get(link, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            tmp = out.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 22):     # 4MB
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r       {done/1e6:6.0f}/{total/1e6:.0f} MB", end="", flush=True)
            print()
            tmp.rename(out)      # atomic: a partial download never looks complete
        print(f"[ok  ] {t}: {out.stat().st_size/1e6:.0f} MB")


def _read_zip_csv(table: str, usecols=None, dtype=None) -> pd.DataFrame:
    """Read the single CSV inside a bulk-export zip."""
    zp = RAW / f"{table}.zip"
    if not zp.exists():
        raise FileNotFoundError(f"{zp} missing — run `sharadar_ingest.py download` first")
    with zipfile.ZipFile(zp) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            return pd.read_csv(fh, usecols=usecols, dtype=dtype, low_memory=False)


def _read_zip_csv_chunked(table: str, usecols, keep_tickers: set,
                          chunksize: int = 2_000_000):
    """Stream a bulk CSV, keeping only rows for `keep_tickers`.

    SEP is ~55M rows; materialising it whole (the ticker column alone is GBs as
    Python objects) would exhaust memory on a laptop. Filtering per chunk keeps
    peak usage to one chunk plus the surviving rows."""
    zp = RAW / f"{table}.zip"
    if not zp.exists():
        raise FileNotFoundError(f"{zp} missing — run `sharadar_ingest.py download` first")
    frames, seen = [], 0
    with zipfile.ZipFile(zp) as z:
        with z.open(z.namelist()[0]) as fh:
            for chunk in pd.read_csv(fh, usecols=usecols, chunksize=chunksize,
                                     low_memory=False):
                seen += len(chunk)
                frames.append(chunk[chunk["ticker"].isin(keep_tickers)])
                print(f"\r      scanned {seen/1e6:.1f}M rows", end="", flush=True)
    print()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _company_key(row, ticker: str) -> str:
    """Mirror the fiscal.ai key format 'EXCHANGE_TICKER' so existing caches,
    the PIT layer and the tracker's snapshot history all still line up.

    `ticker` is passed in rather than read off `row`: the caller indexes the
    metadata BY ticker, so it is the row's index label, not one of its fields."""
    ex = str(row.get("exchange") or "NA").upper().replace(" ", "")
    return f"{ex}_{ticker}"


def build(min_history: int = 260) -> dict:
    """Assemble the artifact dict + write it to data/artifacts/."""
    print("[1/4] tickers …", flush=True)
    meta = _read_zip_csv("TICKERS")
    meta = meta[meta["table"] == "SEP"].copy()          # equities only
    for c in ("firstpricedate", "lastpricedate"):
        if c in meta.columns:
            meta[c] = pd.to_datetime(meta[c], errors="coerce")
    meta = meta.drop_duplicates("ticker", keep="last")
    print(f"      {len(meta):,} equity tickers "
          f"({(meta['isdelisted'] == 'Y').sum():,} delisted)")

    print("[2/4] fundamentals (SF1/ART) …", flush=True)
    # usecols at read time: the full SF1 is ~100 columns x millions of rows.
    sf1 = _read_zip_csv("SF1", usecols=SF1_COLS + ["dimension"])
    sf1 = sf1[sf1["dimension"] == "ART"]                # as-reported TTM ONLY
    keep = [c for c in SF1_COLS if c in sf1.columns]
    sf1 = sf1[keep].copy()
    sf1["datekey"] = pd.to_datetime(sf1["datekey"], errors="coerce")
    sf1 = sf1.dropna(subset=["datekey"]).sort_values(["ticker", "datekey"])
    # Availability date = filing date + the lag censor. Indexing fund_hist by
    # THIS date is what makes the panel point-in-time: a lookup as-of date d
    # can only ever see filings that were already public on d.
    sf1["available"] = sf1["datekey"] + pd.tseries.offsets.BDay(FILING_LAG_DAYS)
    # YoY revenue growth off the TTM series (the growth-mix input), computed
    # per ticker from filings ~4 quarters apart.
    sf1["growth_revenue_1y"] = (sf1.groupby("ticker")["revenueusd"]
                                   .pct_change(periods=4))
    # Derived fundamentals. ART filings are quarterly TTM snapshots, so a 4-step
    # change is year-on-year. pct_change is meaningless when the base is <= 0
    # (a swing from -10 to +5 is not "-150% growth"), so those are dropped
    # rather than allowed to masquerade as extreme growth.
    for src, dst in (("fcf", "growth_fcf_1y"), ("netinc", "growth_netinc_1y")):
        prev = sf1.groupby("ticker")[src].shift(4)
        sf1[dst] = np.where(prev > 0, (sf1[src] - prev) / prev, np.nan)
    rev = sf1["revenueusd"].where(sf1["revenueusd"] > 0)
    sf1["fcf_margin"] = sf1["fcf"] / rev
    sf1["ebitda_margin"] = sf1["ebitda"] / rev
    # Gross debt / EBITDA -- the leverage measure an analyst actually reaches
    # for, and more robust than `de` (debt/equity), which blows up or flips
    # sign when buybacks drive book equity negative. Gross, not NET, of cash:
    # cashneq isn't pulled, and silently calling debt/EBITDA "net debt" would
    # misstate leverage for cash-rich names. Undefined when EBITDA <= 0.
    eb = sf1["ebitda"].where(sf1["ebitda"] > 0)
    sf1["debt_ebitda"] = sf1["debt"] / eb

    # --- the three anomalies with the strongest replication record -----------
    # Novy-Marx (2013) gross profitability: gross profit over ASSETS, not over
    # sales. Gross margin alone just sorts by industry (software ~80%, retail
    # ~25%); scaling by assets asks how much gross profit the balance sheet
    # actually produces, which is comparable across sectors.
    assets = sf1["assets"].where(sf1["assets"] > 0)
    sf1["gp_assets"] = (sf1["grossmargin"] * sf1["revenueusd"]) / assets
    # Sloan (1996) accruals: earnings not backed by cash. Negative is good, so
    # the sign is flipped where it is ranked.
    sf1["accruals"] = (sf1["netinc"] - sf1["ncfo"]) / assets
    # Net share issuance (Daniel-Titman / Pontiff-Woodgate): YoY change in
    # shares outstanding. Dilution is bad, buybacks good -> ranked negatively.
    prev_sh = sf1.groupby("ticker")["sharesbas"].shift(4)
    sf1["share_issuance"] = np.where(prev_sh > 0,
                                     (sf1["sharesbas"] - prev_sh) / prev_sh, np.nan)
    print(f"      {len(sf1):,} ART filings, {sf1['ticker'].nunique():,} tickers")

    print("[3/4] prices (SEP, closeadj) …", flush=True)
    # Only names that could ever clear the model's liquidity floor are worth
    # pricing: edge_lib drops any candidate with no market cap, and the product
    # floor is $2B. Keep everything that EVER reached MCAP_KEEP (well below the
    # floor, so nothing the model could pick is lost) -- this is what makes a
    # 55M-row table tractable, and it discards only names the model would
    # never have selected anyway.
    eligible = set(sf1.loc[sf1["marketcap"] >= MCAP_KEEP, "ticker"].unique())
    eligible &= set(meta["ticker"])
    print(f"      {len(eligible):,} tickers ever >= ${MCAP_KEEP/1e9:.0f}B (of "
          f"{sf1['ticker'].nunique():,} with fundamentals)")
    # The 55M-row scan is by far the slowest step; cache its (small) output so
    # a failure later in the build doesn't cost another full pass.
    cache = RAW / "sep_filtered.pkl"
    sep = None
    if cache.exists():
        try:
            sep = pd.read_pickle(cache)
            print(f"      reusing {cache.name}")
        except Exception as e:
            print(f"      cache unreadable ({e}); rescanning")
    if sep is None:
        sep = _read_zip_csv_chunked("SEP", ["ticker", "date", "closeadj"], eligible)
        # The cache is an optimisation. It must never be able to fail the
        # build -- losing a completed 55M-row scan to a cache write is worse
        # than having no cache at all.
        try:
            sep.to_pickle(cache)
        except Exception as e:
            print(f"      WARN could not cache the scan ({e}); continuing")
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce")
    sep = sep.dropna(subset=["date", "closeadj"])
    print(f"      {len(sep):,} price rows, {sep['ticker'].nunique():,} tickers")

    # Daily point-in-time valuation, resampled to month-end. Full daily
    # resolution for 7k names would add gigabytes to the artifact for no gain:
    # the rebalance clock is 42 trading days, so month-end multiples are at
    # most ~3 weeks stale versus a hold twice that long -- and still far fresher
    # than SF1's as-filed ratios, which can be a full quarter behind.
    daily_by_ticker: dict = {}
    if (RAW / "DAILY.zip").exists():
        print("[3b/4] daily valuation (DAILY) …", flush=True)
        dly = _read_zip_csv_chunked("DAILY", DAILY_COLS, eligible)
        dly["date"] = pd.to_datetime(dly["date"], errors="coerce")
        dly = dly.dropna(subset=["date"]).sort_values(["ticker", "date"])
        dly = (dly.set_index("date").groupby("ticker")
                  .resample("ME").last().drop(columns=["ticker"], errors="ignore"))
        dly = dly.reset_index()
        daily_by_ticker = dict(tuple(dly.groupby("ticker")))
        print(f"      {len(dly):,} month-end rows, {len(daily_by_ticker):,} tickers")
    else:
        print("[3b/4] DAILY.zip absent — falling back to SF1 as-filed multiples")

    print("[4/4] assembling artifact …", flush=True)
    meta_by_ticker = meta.set_index("ticker")
    fund_by_ticker = dict(tuple(sf1.groupby("ticker")))
    out: dict = {}
    n_short = 0
    for tk, g in sep.groupby("ticker", sort=False):
        if tk not in meta_by_ticker.index:
            continue
        if len(g) < min_history:                        # too short to signal on
            n_short += 1
            continue
        m = meta_by_ticker.loc[tk]
        prices = pd.Series(g["closeadj"].values, index=pd.DatetimeIndex(g["date"]),
                           name="close").sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]

        fh = fund_by_ticker.get(tk)
        if fh is not None and len(fh):
            cols = {"calculated_market_cap": fh["marketcap"].values,
                    "growth_revenue_1y": fh["growth_revenue_1y"].values}
            for c in FUND_ATTACH:
                if c in fh.columns:
                    cols[c] = fh[c].values
            fund_hist = pd.DataFrame(cols, index=pd.DatetimeIndex(fh["available"])).sort_index()
            fund_hist = fund_hist[~fund_hist.index.duplicated(keep="last")]
        else:
            fund_hist = pd.DataFrame(
                columns=["calculated_market_cap", "growth_revenue_1y"] + list(FUND_ATTACH),
                index=pd.DatetimeIndex([]))

        # Daily-PIT multiples override the as-filed ones where available: same
        # column names, so downstream code is unchanged and simply gets fresher
        # numbers. Suffixed _d so a screen can tell which source it is using.
        dv = daily_by_ticker.get(tk)
        if dv is not None and len(dv):
            dcols = {f"{c}_d": dv[c].values for c in
                     ("pe", "pb", "ps", "evebitda", "marketcap", "ev") if c in dv.columns}
            dhist = pd.DataFrame(dcols, index=pd.DatetimeIndex(dv["date"])).sort_index()
            dhist = dhist[~dhist.index.duplicated(keep="last")]
            fund_hist = (fund_hist.join(dhist, how="outer").sort_index()
                         if len(fund_hist) else dhist)
            # forward-fill so a rebalance between filings still sees the last
            # known value of each series rather than a hole
            fund_hist = fund_hist.ffill()

        out[_company_key(m, tk)] = {
            "prices": prices,
            "fund_hist": fund_hist,
            "meta": {"ticker": tk,
                     "name": str(m.get("name") or tk),
                     "sector": str(m.get("sector") or "Unknown"),
                     "trading_status": "Inactive" if m.get("isdelisted") == "Y" else "Active"},
        }

    n_dead = sum(1 for b in out.values() if b["meta"]["trading_status"] == "Inactive")
    print(f"      {len(out):,} names kept ({n_dead:,} delisted, "
          f"{n_dead/max(len(out),1)*100:.0f}%); {n_short:,} skipped for <{min_history} bars")

    art = config.ARTIFACT_DIR
    art.mkdir(parents=True, exist_ok=True)
    with open(art / "backtest_data.pkl", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    (art / "meta.json").write_text(
        pd.Series({"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n_names": len(out), "universe_size": len(out),
                   "source": "sharadar"}).to_json(), encoding="utf-8")
    print(f"      wrote {art/'backtest_data.pkl'}")
    return out


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd in ("download", "all"):
        download()
    if cmd in ("build", "all"):
        build()
    if cmd not in ("download", "build", "all"):
        print(f"usage: {argv[0]} [download|build|all]")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
