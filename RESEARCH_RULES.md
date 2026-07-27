# Research rules — read before running anything

These were each learned the expensive way. Breaking one doesn't produce an
error; it produces a number that looks good and is false.

## 1. Measure on the production basis, always

A research script must report the SAME quantity the site will display. If the
product measures on the daily curve, research measures on the daily curve.

Why: research harnesses here computed returns per rebalance — one point every
42 trading days. A series sampled that coarsely cannot see intra-period
troughs, so it reported a **-16% max drawdown for a book whose true drawdown
was -48%**. Nothing errored. Every conclusion drawn from it was wrong, and it
was caught only because the model was run through `run_edge_backtest` before
shipping. Prefer `run_edge_backtest` for any number that will be quoted; if a
bespoke harness is unavoidable, reconcile it against `run_edge_backtest`
BEFORE reporting, and state the basis in the output.

## 2. Never present a research number as a production number

Before any figure is quoted to a human or put on the site, it must come from
the code path that will actually run, with the spec that will actually ship.
Configuration drift is silent: the correlation cap and staggered sleeves cost
~3.8 points of excess that research runs never paid, because research didn't
apply them.

## 3. Don't overfit — and "more tests" is not the definition

Overfitting is choosing a variant *because* it scored well on data you already
looked at, when the choice isn't justified independently of that score.

Legitimate: a small candidate set fixed in advance, chosen on a principled
tiebreak (all-period consistency, an economic mechanism, a prior from the
literature). Picking a 2-month clock from {1m, 2m, 6m} on Sharpe and drawdown
is parameter selection, not overfitting.

Not legitimate: sweeping a grid and reporting the best cell. This repo has run
324-configuration grids where the expected best-of-noise t-stat was **3.40** and
the best real result reached **0.7**. Compute that threshold
(`sqrt(2*ln(N_trials))`) and print it beside any winner.

Rules that follow:
- State the hypothesis and the config list BEFORE running.
- Report in-sample AND out-of-sample side by side, always. A factor that
  sign-flips across the split (ebitda_margin: +0.020 → −0.019) must be shown
  flipping, not summarised away.
- A holdout is single-use. The 2021+ holdout was evaluated across ~40
  configurations; it is contaminated and no longer decides anything.
- A result counts only if neighbouring buckets agree — adjacent eras, related
  factors, or the same effect across independent cuts. One hot cell is noise.
- Report how many names a screen passes. A screen passing 4 names is an
  anecdote, not a strategy.

## 4. Point-in-time discipline is not optional

See `sharadar_kit/CLAUDE.md`. SF1 dimension ART only (MR* are restated and leak
the future), filter on `datekey` not `calendardate`, +1 trading-day filing lag,
`closeadj` for returns, and delisted names STAY IN. The reason this data costs
money is that it is survivorship-free; quietly dropping dead companies inflates
everything — measured here at **+8.5%/yr of fake excess** on the old fiscal.ai
panel, which carried 111 delisted names that all died in 2024 or later.

## 5. Caches are part of the answer

`edge_lib` memoises panels, selection and daily series. Anything that changes
what those produce must be in the cache key. Module globals that change output
(`USE_PIT_UNIVERSE`, `DELIST_HAIRCUT`, `fcf_screen`) are not automatically —
after mutating one, call `edge_lib.reset_caches()`. Clearing `load_edge_panel`
alone is not enough; three caches sit above it. Proven failure: flipping the
universe and clearing only the panel returned the previous variant's numbers
exactly (22.49% both times) with no error.

## 6. Check units and coverage, not just presence

A column that exists can still be meaningless. FCF (dollars) over EV (millions)
gave a yield inflated 1e6×; because ranking is scale-invariant the ordering
looked fine and nothing threw. Print medians and non-null percentages for any
new factor before trusting it.

## 7. Never ship silently

Model parameter changes get shown as a diff and verified through production
before going live. The signal label was hardcoded to "acceleration" in two
places and kept saying so after the signal changed — the site would have
described a model that wasn't running.

## Status (2026-07-27)

- `accel` is FALSIFIED: +0.27%/reb (t=0.4) in-sample, −6.4%/yr out-of-sample,
  −52.8% drawdown, dead in every size / era / sector / regime bucket tested.
- 12-1 momentum is the replacement candidate, conditional on large caps
  (+3.15%/reb, t=2.6), uptrends (+4.06%, t=2.7) and high-dispersion sectors.
- **NOT IN PRODUCTION.** It goes live only when its production-basis numbers
  are verified and acceptable — never on the strength of research-basis ones.
