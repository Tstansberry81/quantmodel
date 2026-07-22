"""Ingest the REAL point-in-time Russell-1000 universe from Norgate (trial: ~2yr).

Pulls, for every current-AND-past R1000 member (delisted included):
  * total-return daily close (splits + dividends), so returns are apples-to-apples
    with the Edge's total-return basis, and
  * the point-in-time index-membership flag (1 on days the name was in the R1000).

Also grabs the S&P 500 level ($SPX) for the 200-day regime and, if available, an
S&P total-return series for the benchmark. Everything is aligned to the S&P trading
calendar and cached to data/cache/norgate_r1000.pkl so the backtest re-runs fast.

This is the survivorship-free universe: names that delisted mid-sample are present,
priced through their delisting, and drop out of membership when they left the index.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
import norgatedata as nd
import config

WATCHLIST = "Russell 1000 Current & Past"
OUT = config.CACHE_DIR / "norgate_r1000.pkl"
TR = nd.StockPriceAdjustmentType.TOTALRETURN
PAD = nd.PaddingType.NONE


def _series(sym, cal, adj=TR):
    p = nd.price_timeseries(sym, stock_price_adjustment_setting=adj,
                            padding_setting=PAD, timeseriesformat="pandas-dataframe")
    if p is None or len(p) == 0:
        return None
    return p["Close"].reindex(cal)


def build():
    # master calendar + S&P from the index database
    spx = nd.price_timeseries("$SPX", padding_setting=nd.PaddingType.ALLMARKETDAYS,
                              timeseriesformat="pandas-dataframe")
    cal = spx.index
    spx_close = spx["Close"]
    # S&P total-return benchmark if the symbol exists, else fall back to $SPX
    sptr = None
    for cand in ("$SPXTR", "$SPX-TR", "$SP500TR"):
        try:
            sptr = nd.price_timeseries(cand, padding_setting=nd.PaddingType.ALLMARKETDAYS,
                                       timeseriesformat="pandas-dataframe")["Close"].reindex(cal)
            print(f"  benchmark: {cand}")
            break
        except Exception:
            continue
    if sptr is None:
        sptr = spx_close
        print("  benchmark: $SPX (no TR series found)")

    syms = nd.watchlist_symbols(WATCHLIST)
    print(f"  {WATCHLIST}: {len(syms)} symbols (current + past). Pulling prices + membership ...")
    price, member, meta = {}, {}, {}
    ok = miss = 0
    for i, s in enumerate(syms):
        try:
            close = _series(s, cal)
            m = nd.index_constituent_timeseries(s, "Russell 1000", timeseriesformat="pandas-dataframe")
            if close is None or m is None:
                miss += 1; continue
            price[s] = close.to_numpy(float)
            member[s] = m["Index Constituent"].reindex(cal).fillna(0).to_numpy(bool)
            try:
                meta[s] = {"name": nd.security_name(s)}
            except Exception:
                meta[s] = {"name": s}
            ok += 1
        except Exception:
            miss += 1
        if (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(syms)}  (ok={ok} miss={miss})")

    data = {"cal": cal, "spx_close": spx_close.to_numpy(float),
            "sptr": sptr.to_numpy(float), "price": price, "member": member, "meta": meta}
    with open(OUT, "wb") as f:
        pickle.dump(data, f)
    ndel = sum(1 for s in price if "-" in s and s.rsplit("-", 1)[-1].isdigit())
    print(f"  DONE: {ok} symbols priced ({ndel} delisted), calendar "
          f"{cal.min().date()}..{cal.max().date()} ({len(cal)} days). -> {OUT}")
    return data


if __name__ == "__main__":
    build()
