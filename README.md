# Quant Model — cross-sectional stock selection + local website

Selects a 7-holding portfolio (6 equities ranked by a z-score composite + a
regime-driven gold sleeve) from the top ~1000 US stocks by market cap, with an
honest point-in-time backtest against the S&P 500 and Nasdaq.

## Data sources
- **fiscal.ai** — universe (`/v2/companies-list`, market-cap sorted) and
  fundamentals (`/v1/company/ratios`).
- **yfinance** — all stock prices/returns on a **total-return** basis (adjusted
  close = splits + dividends reinvested), plus the benchmarks: S&P 500 total
  return (`^SP500TR`), Nasdaq-100 total return (`QQQ`), and gold (`GLD`).
  fiscal.ai prices are split-adjusted only (no dividends), so the price/return
  layer uses yfinance; fiscal.ai remains the source for fundamentals.

## Setup (already done)
```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1) Build the data (run first, and whenever you want fresh data)
```
.venv\Scripts\python.exe build_data.py            # full top-1000 universe
.venv\Scripts\python.exe build_data.py --limit 150   # smaller / faster
```
All fiscal.ai responses are cached under `data/cache/`, so re-runs are fast and
quota-friendly. Artifacts (factor snapshot + price/fundamental history) land in
`data/artifacts/`.

## 2) Run the website
```
.venv\Scripts\python.exe app.py      # -> http://127.0.0.1:5000
```
- **Portfolio page** — the 7 holdings with weights, the HMM market regime,
  a ranked candidate table (returns + excess vs S&P 500 over 3M/6M/12M/2Y/5Y/10Y),
  a stock-vs-S&P 500 selector chart, and an **equation editor** to change any
  threshold or composite weight and recompute live.
- **Backtest page** — growth-of-$1 vs both indexes, decile monotonicity, the IC
  time series, and a performance table (total return, CAGR, Sharpe, max drawdown).

## The model
1. **Universe** — top N US-listed (NYSE/Nasdaq) active names by market cap for
   the live portfolio, plus all available delisted/inactive names added to the
   backtest pool.
2. **Factors** — ROIC, diluted EPS, P/E, FCF margin, revenue growth (fiscal.ai),
   plus momentum 12-1, Sharpe, Sortino, and a low-volatility factor (annualized
   realized vol, lower-is-better) computed from total-return prices. Extra ratios
   (earnings yield, EV/EBITDA, ROE, leverage, shareholder yield…) are pulled and
   ready to weight in.
3. **Scoring** — winsorize (1/99 pct) → sector-neutral z-score → equal-weight
   composite. Weights/thresholds are adjustable in `qmodel/equations.py` or the UI.
4. **Regime** — a Gaussian HMM on S&P 500 weekly returns labels bull/neutral/bear
   and sets the gold sleeve weight (more gold in risk-off).
5. **Backtest** — monthly rebalances over a **point-in-time universe** (top
   `BT_UNIVERSE_SIZE` by market cap *as of each date*, including delisted/inactive
   names), so it holds what was actually large then. Fundamentals lagged 90 days
   to filing date (no lookahead in the signal); IC = Spearman(score, forward 1M
   return). The PIT universe needs `calculated_market_cap` history, which begins
   ~2005, so the backtest window starts there.

## Known limitations (read these)
- **Survivorship bias (reduced, not eliminated)**: the backtest now uses a
  point-in-time market-cap universe and includes delisted/inactive names — so it
  no longer just holds today's winners. BUT fiscal.ai carries only ~211 inactive
  names (mostly recent M&A); long-dead companies (Enron, Lehman, etc.) are
  absent, so absolute returns are still somewhat optimistic. Fully eliminating it
  needs a delisted-history dataset (CRSP, Norgate, etc.). IC and decile
  monotonicity remain the most trustworthy signal-quality measures.
- **Gold overlay lookahead**: the HMM is fit on the full sample, so the gold
  sleeve in the backtest carries mild lookahead. The equity signal (IC/deciles)
  is strictly point-in-time.
- **Returns are total return** (dividends reinvested), measured against
  total-return benchmarks (`^SP500TR`, `QQQ`) for a fair comparison.
- **Annual fundamentals**: currently uses annual periods; LTM/quarterly is a
  straightforward upgrade.

## Tuning
Everything adjustable lives in `qmodel/equations.py` (`FACTORS`, `EXTRA_RATIOS`,
`SETTINGS`) and `config.py` (universe size, intervals, gold weights, throttle).
