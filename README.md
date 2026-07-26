# The Edge — short-horizon momentum model + site

This repo is the in-house model and viewer for **The Edge**: a short-horizon,
market-data-only momentum strategy on a Russell-1000-proxy universe, with a
point-in-time backtest, a live paper-trading tracker, and an export pipeline
that feeds the consumer-facing **Vision** static site (`../vision`).

The long-term fundamentals model ("Slow Burn") has been removed; `edge_data.py`
is the self-contained data layer that replaced its `qmodel` package.

## The model (product spec, `edge_lib.EDGE_SPEC`)

- **Signal** — *acceleration*: the 3-month return minus the prior 3-month
  return, z-scored cross-sectionally at each rebalance.
- **Universe** — top ~1000 US names by point-in-time market cap, $2B floor.
  `pit_universe.py` swaps in real PIT membership (Norgate or a CSV) when present.
- **Selection** — greedy best-first under a 0.50 correlation cap (126-day
  lookback); 75% of the basket drawn from the ≥15% YoY revenue-growth pool
  (`growth_mix`); top **10** equal-weight.
- **Clock** — every 42 trading days (~2 months), run as **two staggered sleeves**
  offset by half a period, so results don't hinge on rebalance-date luck.
- **Regime** — the 200-day-MA de-risk is evaluated **daily** across the hold
  window (`continuous_regime`), cutting to 25% invested rather than riding a
  fast intra-window crash through to the next rebalance.
- **Execution lag** — the signal is formed on close *t*, the trade enters at
  close *t+1*. You cannot compute the whole cross-section and trade the close it
  came from.
- **Costs** — 10 bps × turnover, deducted per rebalance.

## Setup (macOS)

```
python3 -m venv .venv-mac
.venv-mac/bin/pip install -r requirements.txt
```

## Data

Prices and fundamentals come from a prebuilt artifact bundle
(`data/artifacts/backtest_data.pkl` + caches), read through `edge_data.py`:

```
.venv-mac/bin/python fetch_data.py        # download+extract the bundle from $DATA_URL
.venv-mac/bin/python make_data_bundle.py  # repackage local artifacts -> data_bundle.zip
```

Supporting builders (each needs `FISCAL_API_KEY` where noted):

- `fiscal_fundamentals.py` — canonical standardized financials from fiscal.ai
  (Novy-Marx gross profitability, Piotroski, Sloan accruals…).
- `fiscal_quarterly_eps.py` — quarterly EPS surprise, for the PEAD sleeve.
- `build_ohlc.py` — adjusted OPEN alongside CLOSE, for overnight/intraday tests.
- `norgate_ingest.py` — real point-in-time Russell-1000 membership + delisted
  names from Norgate; `norgate_devalidate.py` re-checks the de-bias claim.

Secrets live in a gitignored `.env` at the repo root (`FISCAL_API_KEY`,
`ANTHROPIC_API_KEY`, optionally `GITHUB_TOKEN`); importing `config` loads it for
every entrypoint. On Render, set them as real environment variables.

## Run the site

```
.venv-mac/bin/python app.py      # -> http://127.0.0.1:5000
```

- **The Edge** (`/edge`) — backtest dashboard: growth-of-$1 vs S&P/Nasdaq,
  per-window performance (1Y–20Y/MAX), turnover, Sharpe/Sortino/drawdown.
  Controls: window, clock, growth mix, basket size, dollar amount.
- **Edge Tracker** (`/edge-tracker`) — the live paper-trading record: the
  current book, the rebalance-by-rebalance log vs the S&P, hit-rate stats, and
  the forward-accruing snapshot record in `data/cache/edge_tracker.json`.
- **Model** (`/model`) — the full equation spec.
- **APIs** — `/api/edge_backtest`, `/api/edge_tracker`, `/api/meta`,
  `/api/sync_vision` (publishes to Vision), `/api/chat` (in-app explainer;
  needs `ANTHROPIC_API_KEY`). Compute endpoints are cached across the whole
  selector space, rate-limited per IP, and warmed on boot.

## Vision export (consumer product)

```
.venv-mac/bin/python export_vision.py [--window 2Y --hold 21 --mix 0.75]
```

Writes `window.VISION_DATA` (current book + primary curve + 20-year track
record + headline stats) to `../vision/vision_data.js`. The in-app **Sync to
Vision** button calls the same code and pushes straight to the Vision repo when
`GITHUB_TOKEN` is set, so Render redeploys the product site.

## Deployment

Render web service (see `render.yaml`, `DEPLOY.md`): gunicorn, 2GB plan, bundle
fetched at build time from `DATA_URL`.

## Known limitations (read these)

- **Survivorship bias** — the proxy universe only contains names the artifact
  cache knows, and its delisted coverage is thin, so absolute backtest returns
  are optimistic. See `Edge_Survivorship_Validation.pdf` and
  `build_survivorship_report.py`; the Norgate de-bias result was **retracted**
  after rebuilding on the production basis, so the size of the bias is currently
  **inconclusive** — not "measured and small".
- **Short windows are lumpy** — few rebalances; judge on long windows and on the
  tracker's forward, out-of-sample record.
- **Momentum-regime risk** — acceleration fades in choppy, rotational tapes; the
  200dMA overlay is the only brake.

Backtests are net of ~10bps costs, total-return vs total-return benchmarks.
Not investment advice.
