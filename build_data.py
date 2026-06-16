"""One-shot (resumable) data build.

Pulls the universe, then per-name fundamentals + prices from fiscal.ai (all
cached to disk), assembles the live factor snapshot, and pickles the per-name
price + fundamentals history that the backtest needs. Also warms the yfinance
benchmark/gold cache.

Run:  python build_data.py            # full universe (config.UNIVERSE_SIZE)
      python build_data.py --limit 150
      python build_data.py --resume    # skip names already in artifacts
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time

import pandas as pd

import config
from qmodel import universe as uni, factors as factmod, fiscal, market
from qmodel.equations import all_ratio_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=config.UNIVERSE_SIZE)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] Building universe (top {args.limit} US by market cap)...")
    members = uni.build_universe(size=args.limit)
    uni.save(members)
    print(f"      {len(members)} names.")

    print("[2/4] Warming benchmark + gold cache + total-return prices (yfinance)...")
    try:
        bm = market.benchmarks()
        print("      benchmarks: " + ", ".join(f"{k}:{len(v)}" for k, v in bm.items()))
    except Exception as e:
        print("      WARN benchmarks failed:", e)
    tickers = [m["ticker"] for m in members]
    tr_prices = market.universe_total_return_prices(tickers)
    print(f"      total-return price series: {len(tr_prices)}/{len(tickers)}")
    market_series = market.get_prices(config.BENCH_SP500)   # for residual momentum

    print(f"[3/4] Fetching fundamentals (fiscal.ai) for {len(members)} names...")
    factor_rows = []
    bt_data = {}
    rids = all_ratio_ids()
    n_fallback = 0
    for i, m in enumerate(members, 1):
        ck = m["company_key"]
        try:
            rj = fiscal.ratios(ck, rids, period_type="annual")
            prices = tr_prices.get(m["ticker"])
            if prices is None or len(prices) == 0:   # fall back to fiscal split-adj prices
                prices = factmod.price_series(ck); n_fallback += 1
            row = factmod.build_factor_row(m, prices=prices, ratio_json=rj, market=market_series)
            if row:
                factor_rows.append(row)
            bt_data[ck] = {
                "prices": prices,
                "fund_hist": factmod.fundamentals_history(rj or {}),
                "meta": {"ticker": m["ticker"], "name": m["name"], "sector": m["sector"],
                         "trading_status": m.get("trading_status", "Active")},
            }
        except Exception as e:
            print(f"      WARN {ck}: {e}")
        if i % 50 == 0 or i == len(members):
            print(f"      {i}/{len(members)}  ({time.time()-t0:.0f}s)")
    print(f"      ({n_fallback} names used fiscal price fallback)")

    print("[4/4] Saving artifacts...")
    # JSON-safe snapshot (returns dict nested per row)
    (config.ARTIFACT_DIR / "factors.json").write_text(
        json.dumps(factor_rows, default=_jsonsafe), encoding="utf-8")
    with open(config.ARTIFACT_DIR / "backtest_data.pkl", "wb") as f:
        pickle.dump(bt_data, f)
    meta = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_names": len(factor_rows), "universe_size": len(members)}
    (config.ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {time.time()-t0:.0f}s. {len(factor_rows)} factor rows, "
          f"{len(bt_data)} price/fundamental series.")


def _jsonsafe(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except Exception:
        pass
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
