"""One-time fetch of daily OPEN + CLOSE (split+dividend adjusted) for the Edge
universe, so the overnight/intraday decomposition (Lou-Polk-Skouras) can be tested.

Our existing cache stores adjusted CLOSE only. This adds adjusted OPEN. Both come
from the same yfinance auto_adjust=True download, so overnight vs intraday returns
are on a consistent basis. Caches to data/cache/ohlc.pkl keyed by company_key:
    {company_key: DataFrame[open, close] indexed by date}

Caveat: auto_adjust puts the dividend drop into the overnight leg on ex-dates (a
known second-order artifact vs CRSP's careful handling) -- fine for an IC screen.

Run: .venv/Scripts/python.exe build_ohlc.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, time
import pandas as pd
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine + market)
market = engine             # yahoo_symbol now lives in edge_data

OUT = os.path.join("data", "cache", "ohlc.pkl")


def main():
    data = engine._load_bt_data()
    # map company_key -> yahoo symbol via the stored ticker
    ck_by_ysym, tickers = {}, {}
    for ck, b in data.items():
        tk = b.get("meta", {}).get("ticker")
        if not tk:
            continue
        ys = market.yahoo_symbol(tk)
        ck_by_ysym[ys] = ck
        tickers[ck] = ys
    ysyms = list(ck_by_ysym.keys())
    print(f"Fetching OHLC for {len(ysyms)} names (batched) ...")

    import yfinance as yf
    out, chunk, t0 = {}, 150, time.time()
    for i in range(0, len(ysyms), chunk):
        batch = ysyms[i:i + chunk]
        try:
            df = yf.download(batch, period="max", interval="1d", progress=False,
                             auto_adjust=True, group_by="ticker", threads=True)
        except Exception as e:
            print(f"  batch {i} failed: {e}"); continue
        for ys in batch:
            try:
                sub = df[ys] if len(batch) > 1 else df
                o, c = sub["Open"].dropna(), sub["Close"].dropna()
                idx = o.index.intersection(c.index)
                if len(idx) < 300:
                    continue
                frame = pd.DataFrame({"open": o.reindex(idx), "close": c.reindex(idx)})
                frame.index = pd.to_datetime(frame.index)
                out[ck_by_ysym[ys]] = frame
            except Exception:
                continue
        print(f"  {min(i+chunk,len(ysyms))}/{len(ysyms)}  ({time.time()-t0:.0f}s, kept {len(out)})")
    if out:
        pd.to_pickle(out, OUT)
        print(f"Saved {len(out)} names -> {OUT}")
    else:
        print("No data fetched.")


if __name__ == "__main__":
    main()
