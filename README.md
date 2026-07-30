# The Edge — short-horizon momentum model + site

This repo is the in-house model and viewer for **The Edge**: a short-horizon,
market-data-only momentum strategy on a Russell-1000-proxy universe, with a
point-in-time backtest, a live paper-trading tracker, and an export pipeline
that feeds the consumer-facing **Vision** static site (`../vision`).

The long-term fundamentals model ("Slow Burn") has been removed; `edge_data.py`
is the self-contained data layer that replaced its `qmodel` package.

## The model (product spec, `edge_lib.EDGE_SPEC`)

Full lineage, with what each change cost and what was rejected alongside it, is
in **[MODEL_VERSIONS.md](MODEL_VERSIONS.md)**. Current shipped version: **v5**.

- **Signal** — *12-1 momentum* (Jegadeesh–Titman): the trailing 12-month return
  **skipping the most recent month**, z-scored cross-sectionally at each
  rebalance. Acceleration was retired 2026-07-27 after being falsified on
  survivorship-free data.
- **Universe** — top ~1000 US names by point-in-time market cap, **$10B floor**.
  The floor is load-bearing: momentum degrades monotonically as it drops.
- **Solvency screens** — trailing **FCF margin > 0** and **debt/EBITDA ≤ 4**.
  The only fundamental screens that improved return *and* risk; valuation caps
  (P/E) cost Sharpe and quality screens did nothing for drawdown.
- **Selection** — top **10** equal-weight, at most **2 names per sector**.
  The correlation cap is OFF (it pushed the book away from the signal).
- **Clock** — the **first trading day of every month**. Each book is held until
  the next replaces it, so windows tile the calendar exactly rather than running
  a fixed day-count. Set `rebal_months=2` for the 2-month clock.
- **Regime** — the 200-day-MA de-risk is evaluated **daily** across the hold
  window (`continuous_regime`), cutting to 25% invested.
- **Volatility target** — exposure scaled toward **25% annualized** on the
  book's own trailing 21-day volatility, causal (`.shift(1)`), de-lever only.
- **Execution lag** — signal formed on close *t*, trade enters at close *t+1*.
  You cannot compute the whole cross-section and trade the close it came from.
- **Costs** — 10 bps × turnover, including the turnover of re-levering.

Backtest, MAX window, net of costs: **17.11%/yr, −28.6% max drawdown**, against
the S&P's 8.71% and −55.25%. That is a backtest — the forward record on the Edge
Tracker page is the only out-of-sample evidence.

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
optionally `GITHUB_TOKEN` / `RENDER_API_KEY`); importing `config` loads it for
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
  `/api/sync_vision` (publishes to Vision), `/api/diag` (panel fingerprint vs
  what production expects, plus thread stacks — the fastest way to answer "why
  is it slow"). Compute endpoints are cached across the whole selector space,
  rate-limited per IP, serialized by one compute lock, and warmed on the first
  request (never at import: a lock held across gunicorn's fork deadlocks the
  worker).

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
