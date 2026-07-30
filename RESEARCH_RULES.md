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
- **Wiggle the threshold.** For any cutoff, sweep it and look at the SHAPE. A
  real mechanism gives a smooth response — `vol_target` at 25/20/15/12% traces a
  monotone drawdown curve, which is the reason that lever is trusted. A fitted
  number gives a peak at the value you happened to try first with worse
  neighbours on both sides. The hypergrowth cap died here: excluding YoY revenue
  growth above 50% beat the base on every axis, but every looser cap
  (75/100/150/200%) was *worse than base on CAGR* and 30% collapsed again. No
  mechanism is harmless at 100%, decisive at 45%, and harmless at 30%.
- **Count sub-eras, then check their MAGNITUDES.** "Better in 4 of 4 eras" is
  worth little if three move by 0.1–0.8pts and the fourth by 7.6pts — that is
  one episode sliced into buckets, not four agreements. The solvency screens
  passed this properly (5.1pts in 2013-19 *and* 17.1pts in 2020-26); the
  hypergrowth cap did not, while passing the naive count.

## 4. Point-in-time discipline is not optional

See `sharadar_kit/CLAUDE.md`. SF1 dimension ART only (MR* are restated and leak
the future), filter on `datekey` not `calendardate`, +1 trading-day filing lag,
`closeadj` for returns, and delisted names STAY IN. The reason this data costs
money is that it is survivorship-free; quietly dropping dead companies inflates
everything — measured here at **+8.5%/yr of fake excess** on the old fiscal.ai
panel, which carried 111 delisted names that all died in 2024 or later.

**Survivorship-free DATA is necessary but not sufficient** — the panel can
reintroduce the bias by itself. Audited 2026-07-29:

* The current artifact is clean: 7,826 delisted names (64% of it), spread evenly
  across every year; 590 of them once reached $10B, at 12–38/year with no
  clustering. The model demonstrably eats real losses (worst positions −63%,
  −62%, −62%; five of the ten worst later delisted).
* But the panel USED to drop a name on date *d* because it knew the company
  would stop trading within the next 42 days. That is look-ahead introduced by
  panel construction, not by the vendor. Fixed — `DELIST_HAIRCUT` now defaults
  to 0.0 — at a cost of 0.18pts of CAGR (18.47 → 18.29%).
* Still true, and bounded: `daily_return_matrix()` forward-fills prices, so a
  delisted name is frozen at 0% return on the daily curve rather than marked
  down. Right for an acquisition, generous for a bankruptcy by the final gap.
  Affects 1.1% of positions — all acquisitions in this history.

Two lessons. **Audit the panel-construction path, not just the vendor.** And
**when a stress knob shows no effect, prove the knob reaches the path you are
measuring** — the delist haircut moves `fwd_ret` (per-rebalance) while the site
reports the daily curve, so "−100% wipeout changes nothing" looked like a clean
pass and was actually measuring the wrong code path.

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

## 6b. A knob with no effect is a bug until proven otherwise

Recurring failure, hit twice now. The delisting stress test returned identical
CAGR at −30%, −50% and −100% haircuts; a −100% wipeout cannot change nothing, and
the cause was that the knob only reached `fwd_ret` while the measurement came off
the daily curve. Then, sweeping the rebalance grid's phase, six offsets returned
byte-identical rows to four decimal places — because `_edge_daily` takes no
offset argument at all (it derives offsets from `stagger`), so the sweep was six
copies of one run.

Both were caught by the same reflex, and it is the rule: **before reporting that
a lever does nothing, prove the lever reached the code path being measured.**
Vary it and demand the number move somewhere. Identical to four decimals is not
a finding, it is a disconnected wire.

## 7. Never ship silently

Model parameter changes get shown as a diff and verified through production
before going live. The signal label was hardcoded to "acceleration" in two
places and kept saying so after the signal changed — the site would have
described a model that wasn't running.

## 8. An arbitrary implementation choice is a free parameter — sweep it

**Found 2026-07-30, and it invalidated a headline number that had been on the
site for two days.**

The rebalance grid was built as `monthly[:-hold][::hold]` — a fixed stride of
`hold` trading days starting from whatever day happened to sit 400 calendar days
into the sample. Nobody chose that start date; it fell out of a lookback buffer.
It was never treated as a parameter, so it was never swept.

Sliding that anchor a week at a time, holding everything else fixed:

| phase | CAGR | Sharpe | maxDD | worst drawdown |
|------:|-----:|-------:|------:|----------------|
| 0  | 18.29% | 0.934 | **−30.76%** | 2021-08 → 2022-01 |
| 7  | 16.95% | 0.874 | −43.62% | 2021-08 → 2023-10 |
| 14 | 17.08% | 0.884 | −43.21% | 2021-02 → 2023-10 |
| 21 | 15.79% | 0.826 | −44.79% | 2021-08 → 2023-10 |
| 28 | 15.68% | 0.824 | −47.79% | 2021-02 → 2023-10 |
| 35 | 16.33% | 0.850 | −48.15% | 2021-06 → 2023-10 |

Phase 0 — the shipped one — is the **best of six on every axis**, and it is not
marginally best: the drawdown spread is 17.4 points and the median is −44.21%.
Look at the last column. Every other phase stays underwater until October 2023;
phase 0 recovers in January 2022. It exited the November-2021 momentum peak on a
lucky week and re-entered on another. That is the entire −30.8%.

So `−30.8% max drawdown`, `18.29% CAGR` and `Sharpe 0.934` were never properties
of the model. They were properties of one arbitrary start date, and they were
published on the site, written into `EDGE_SPEC`'s comments as the measured effect
of the volatility target, and quoted in `MODEL_STATUS`.

The rule: **if a number falls out of an implementation detail nobody deliberately
chose, it is a free parameter you have already fitted without noticing.** Sweep
it. Report the median and the spread, not the run you happened to write first.
Structural choices of this kind — grid phase, tie-breaking order, which bar a
window starts on, how a partial period is handled — do not announce themselves as
parameters, which is exactly why they escape every overfitting check in rule 3.

Corollary: the fix is not to hunt for a better phase. It is to stop having one.
The grid is now anchored to calendar month starts, so the phase is fixed by the
calendar rather than by an accident, and the reported numbers are what that
convention actually produces.

## 9. A forward record's timestamp is the evidence — the book date is just a label

Discovered 2026-07-30, two trading days before it would have published a fake
number.

The v5 book is *dated* 2026-07-01, because that is the rebalance the panel
produced it for. It was first written down on 2026-07-30, when the solvency
screens shipped — by which point ~91% of its 07-01 → 08-03 window had already
happened. On 3 August the row would have flipped to CLOSED and posted the full
window's return inside the panel headed *"the only out-of-sample evidence."*

Every individual number would have been correct. The basket was real, the dates
were real, the return was real. The claim built on them — that this was
out-of-sample — was false, because the selection was made with the outcome
already visible.

The rule: **a rebalance is forward evidence only if the wall-clock moment it was
recorded precedes the window it is scored over.** Store that timestamp, compare
it to the book date, and refuse to score anything where the gap is material.
Nothing about a backfilled row looks wrong from the inside; the only thing that
distinguishes it from genuine evidence is a timestamp, so the timestamp has to
be load-bearing rather than decorative.

Where this hides: any time the rules change mid-period. Re-picking a book under
new rules re-dates it to the current rebalance, which silently backdates its
entry. The more careful you are about applying new rules consistently to
history, the more likely you are to walk into this.

Corollary — the same defect had a second face. Exit dates were being
approximated (`hold × 7/5` calendar days) rather than read from the panel's own
`next_dates`. That is correct for a fixed trading-day stride and wrong for a
month grid, where gaps run 15–23 trading days. It put the 07-01 book's exit at
07-30 instead of 08-03, which is what made a row appear to open and close on the
same day. **Derive a date only when you cannot read it** — the panel already
knew every exit, because tiling the calendar exactly is its construction rule.

## Status (2026-07-27)

- `accel` is FALSIFIED: +0.27%/reb (t=0.4) in-sample, −6.4%/yr out-of-sample,
  −52.8% drawdown, dead in every size / era / sector / regime bucket tested.
- 12-1 momentum is the replacement candidate, conditional on large caps
  (+3.15%/reb, t=2.6), uptrends (+4.06%, t=2.7) and high-dispersion sectors.
- **NOT IN PRODUCTION.** It goes live only when its production-basis numbers
  are verified and acceptable — never on the strength of research-basis ones.
