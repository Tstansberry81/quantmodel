"""The Edge -- MARKET-TIMING / regime-overlay test (round-3 market phenomena).

Tests the market-phenomena shortlist that act as EXPOSURE/TIMING overlays: does a
better market-timing rule improve the concentrated book's Sharpe/drawdown versus
the current 200-day-MA regime filter? The accel BOOK is held fixed; only the
exposure rule e_t (fraction in the book vs cash) changes. This isolates the
timing overlay from stock selection.

Rules tested (all causal: signal known at prior close, applied next day):
  * PRODUCT       -- the shipped rule: per-rebalance 200dMA -> 0.25 exposure when
                     S&P < 200dMA (regime_expo). The honest baseline to beat.
  * always-on     -- no timing (100% book always). Reference upper/again vol.
  * d200_25/_cash -- 200dMA applied DAILY (to 0.25, or full cash)
  * sma10mo       -- Faber 2007: in when S&P > 10-month (~210d) SMA, else cash
  * tsmom12_1     -- Moskowitz-Ooi-Pedersen: in when 12-1 month S&P return > 0
  * voltgt_mkt    -- Moreira-Muir style: scale by min(1, target/realized S&P vol)
  * voltgt_book   -- vol-target on the BOOK's own trailing vol (most faithful M-M)
  * vrp           -- Bollerslev-Tauchen-Zhou: de-risk when variance risk premium
                     (VIX^2 - realized var) < 0 (a realized-vol shock)

Exposure changes cost TC_BPS per unit turnover/day (SPY is cheap) -- this is the
Moreira-Muir cost caveat made real. Metrics on the daily curve (true maxDD).

Usage: .venv/Scripts/python.exe edge_test_timing.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import config
import edge_lib as E
import edge_data as D

S = E.EDGE_SPEC
RF_D = config.RISK_FREE_ANNUAL / 252.0
TC = 1e-4                      # 1bp per unit daily exposure change (index/cash is cheap)


def book_daily(n, regime_expo):
    """Daily NET returns of the staggered accel book. regime_expo=1.0 -> pure book
    (no de-risk); 0.25 -> the shipped product rule."""
    idx, model, spx, ndx, turn, holds = E._edge_daily(
        S["hold"], n, S["mcap_floor"], S["corr_cap"], S["corr_lookback"],
        regime_expo, S["cost_bps"], (("accel", 1.0),),
        S["growth_mix"], S["growth_thresh"], S["stagger"])
    return pd.Series(model, index=idx), pd.Series(spx, index=idx)


def market_exposures(idx):
    """Causal daily exposure e_t in [0,1] for each market-based timing rule,
    aligned to the book's daily index."""
    spx = D.benchmarks()["SP500"].dropna()
    vix = D.get_prices("^VIX").dropna()
    r = spx.pct_change()
    sma200 = spx.rolling(200).mean()
    sma210 = spx.rolling(210).mean()
    mom = spx.shift(21) / spx.shift(252) - 1                 # 12-1 month trend
    rv = r.rolling(21).std() * np.sqrt(252)                  # realized ann vol
    rv_tgt = rv.rolling(252).median()                        # causal target
    impvar = (vix / 100.0) ** 2
    realvar = (rv ** 2).reindex(vix.index, method="ffill")
    vrp = (impvar - realvar).reindex(spx.index, method="ffill")

    raw = {
        "always":     pd.Series(1.0, index=spx.index),
        "d200_25":    pd.Series(np.where(spx > sma200, 1.0, 0.25), index=spx.index),
        "d200_cash":  pd.Series(np.where(spx > sma200, 1.0, 0.0), index=spx.index),
        "sma10mo":    pd.Series(np.where(spx > sma210, 1.0, 0.0), index=spx.index),
        "tsmom12_1":  pd.Series(np.where(mom > 0, 1.0, 0.0), index=spx.index),
        "voltgt_mkt": np.minimum(1.0, rv_tgt / rv),
        "vrp":        pd.Series(np.where(vrp > 0, 1.0, 0.25), index=spx.index),
    }
    out = {}
    for k, s in raw.items():
        s = pd.Series(np.asarray(s, float), index=spx.index).shift(1)   # causal
        out[k] = s.reindex(idx, method="ffill").clip(0.0, 1.0)
    return out


def book_voltarget(b):
    """Vol-target on the book's OWN trailing vol (most faithful Moreira-Muir),
    causal, capped at 1.0 (long-only, no leverage)."""
    rvb = b.rolling(21).std() * np.sqrt(252)
    tgt = rvb.expanding(min_periods=126).median()
    e = np.minimum(1.0, tgt / rvb).shift(1)
    return e.reindex(b.index).clip(0.0, 1.0)


def metrics(port):
    r = np.nan_to_num(np.asarray(port, float))
    cagr, dd, sh = E._perf_daily(r)
    so = E.sortino(r, ppy=252.0)
    return cagr, sh, so, dd


def apply_rule(b, e):
    e = e.reindex(b.index).ffill().fillna(1.0)
    de = e.diff().abs().fillna(0.0)
    port = e.values * b.values + (1.0 - e.values) * RF_D - TC * de.values
    return metrics(port), float(e.mean())


def main():
    for n in (7, 10):
        print("\n" + "=" * 82)
        print(f"MARKET-TIMING OVERLAY on the accel book -- n={n}, hold={S['hold']}, staggered/daily")
        print("=" * 82)
        print(f"{'rule':<14}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}{'avg expo':>10}")
        print("-" * 82)

        # references straight from the product machinery
        prod, _ = book_daily(n, S["regime_expo"])          # shipped per-rebalance 200dMA@0.25
        pure, _ = book_daily(n, 1.0)                        # pure book, no de-risk
        for lab, series in (("PRODUCT", prod), ("always-on", pure)):
            c, sh, so, dd = metrics(series.to_numpy())
            print(f"{lab:<14}{c*100:>7.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%{1.0:>10.2f}")

        # timing overlays applied to the PURE book
        exps = market_exposures(pure.index)
        exps["voltgt_book"] = book_voltarget(pure)
        order = ["d200_25", "d200_cash", "sma10mo", "tsmom12_1",
                 "voltgt_mkt", "voltgt_book", "vrp"]
        base_sh = metrics(prod.to_numpy())[1]; base_dd = metrics(prod.to_numpy())[3]
        for k in order:
            (c, sh, so, dd), ex = apply_rule(pure, exps[k])
            tag = "  <-- beats PRODUCT" if (sh > base_sh + 1e-9 and dd >= base_dd - 1e-9) else ""
            print(f"{k:<14}{c*100:>7.1f}%{sh:>8.2f}{so:>9.2f}{dd*100:>7.0f}%{ex:>10.2f}{tag}")
        print("-" * 82)
        print("PRODUCT = shipped 200dMA rule (baseline to beat). A rule 'beats' only if Sharpe")
        print("improves AND maxDD is no worse (drawdown priority). avg expo = mean fraction in book.")


if __name__ == "__main__":
    main()
