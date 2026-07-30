# The Edge — model version log

Every change to `EDGE_SPEC` that altered what the model holds or what it
reports, newest first, with the numbers measured **on the production daily
basis** at the time (`run_edge_backtest`, MAX window, net of costs).

Why this file exists: the spec's inline comments explain *why* each parameter is
what it is, but they are overwritten when a parameter changes — so the history
of what we believed, and what it cost to find out, was being lost. Several
figures quoted on the public site survived for days after the model that
produced them had been replaced. A version log is the cheapest defence against
quoting a dead model.

**Read the caveat at the bottom before treating any row as a track record.**

---

## v5 — solvency screens · 2026-07-30

| | CAGR | Sharpe | Sortino | maxDD | vs v4 |
|---|---|---|---|---|---|
| **v5** | **17.11%** | 0.910 | 1.303 | **−28.58%** | −0.37pts CAGR, **+17.1pts drawdown** |
| v4 | 17.48% | 0.900 | 1.283 | −45.69% | |
| S&P over the same window | 8.71% | 0.531 | — | −55.25% | |

**Changed:** added `fcf_positive=True` and `debt_ebitda_max=4.0` — the first
fundamental screens this model has ever shipped.

**Why:** the worst drawdown is a momentum *unwind*, and the names that fall
hardest in one are those that cannot fund themselves when financing dries up.
That is a solvency story, not a valuation one.

**What was rejected alongside it** (ten screens tested, so this list matters as
much as the winner): P/E ≤ 40 and cheapest-70%-by-P/E both cost Sharpe in every
form. Low accruals gave no drawdown benefit. **GP/assets — paper-tracked for
weeks as a validated upgrade — came out worse than shipped on every axis.**
"No net share issuance" cut drawdown hardest (−24.5%) but cost 2.9pts of CAGR
and 0.05 Sharpe, and removed 47% of the universe.

**Evidence it is not just the best of ten:**
- Better in **4 of 4** independent sub-eras: −23.5 / −19.2 / −24.0 / −28.6%
  against v4's −25.6 / −20.9 / −29.1 / −45.7%.
- Still additive after volatility targeting: at `vol_target=0.15` it takes the
  drawdown from −28.7% to −21.6%.
- **Dominates de-levering.** Dropping `vol_target` to 0.15 reaches the same
  −28.7% drawdown but gives up 3.5pts of CAGR (13.96%). Cutting exposure
  throttles the winners too; removing insolvent names does not.
- `fcf_margin` is 100% populated in every era, so the screen is never a
  data-availability filter wearing a fundamental label (RESEARCH_RULES #6).
- The book still fills 10 names across ≥5 sectors at **every** rebalance.

**Not yet confirmed out of sample.** Only the forward record can do that.

### Re-measured on v5 — survivorship exposure CHANGED SIGN (2026-07-30)

| | CAGR | Sharpe | maxDD |
|---|---|---|---|
| shipped (dying names held to last trade) | 17.11% | 0.910 | −28.58% |
| look-ahead (dying names excluded) | 16.92% | 0.900 | −28.95% |

**−0.19 pts.** Excluding dying names makes the model WORSE, so there is no
residual survivorship inflation left — the delisting treatment runs in the
model's favour.

On v3 the sign was the other way (the look-ahead variant was 0.18pts *higher*).
The solvency screens flipped it: `fcf_positive` and `debt_ebitda_max` remove the
distressed companies that die badly, so the names that still delist are
overwhelmingly acquisitions, and holding those to their last trade captures the
deal premium. Affected positions: 20 of 3,300 (0.6%).

The site previously said "residual survivorship exposure is about 0.2 percentage
points of CAGR", which reads as 0.2pts of inflation — correct for v3, backwards
for v5. Corrected in `static/js/edge.js`.

**KNOWN STALE, deliberately not yet changed:** the `DELIST_HAIRCUT` comment in
`edge_lib.py` still quotes the v3 figures (18.47% → 18.29%). Editing that file
changes the panel fingerprint and invalidates the shipped bundle, so it is
batched for the next rebuild+republish rather than triggering one for a comment.

### Tested after v5 and REJECTED — revenue-growth gates (2026-07-30)

`rev_growth` is 96% populated and stored as a fraction (median +7.9% YoY).

**Requiring growth does not work**, at any bar, on top of v5:

| gate | CAGR | Sharpe | maxDD |
|---|---|---|---|
| v5 base | 17.11% | 0.910 | −28.58% |
| rev growth > 0 | 15.21% | 0.835 | −28.63% |
| rev growth ≥ 15% | 14.69% | 0.805 | −29.49% |
| rev growth ≥ 25% | 13.20% | 0.744 | −39.49% |
| net income growth > 0 | 12.90% | 0.748 | −32.36% |

This reproduces v2's `growth_mix` rejection, but as a GATE, on the current
signal, grid and base — a much stronger result than the original. Mechanically
it is near-redundant: `fcf_positive` already removes most of what "growing
healthily" proxies for, and the gate pays for the overlap in lost return. The
≥25% bar also failed to fill 10 names at 9 rebalances, so part of that row is
the sector cap back-filling rather than the screen working.

**Excluding hypergrowth looked excellent and still failed.** Capping YoY revenue
growth at 50% gave 17.92% / 0.967 / −23.43% — better than v5 on every axis. It
was rejected on the threshold response:

| cap | CAGR | Sharpe | maxDD |
|---|---|---|---|
| none | 17.11% | 0.910 | −28.58% |
| >200% | 16.43% | 0.884 | −27.67% |
| >150% | 16.18% | 0.873 | −27.12% |
| >100% | 16.40% | 0.885 | −26.44% |
| >75% | 16.78% | 0.906 | −22.75% |
| **>50%** | **17.92%** | **0.967** | −23.43% |
| >40% | 18.16% | 0.986 | −22.65% |
| >30% | 16.53% | 0.927 | −24.99% |

Every loose cap is WORSE than base on CAGR; the gain exists only in a narrow
40–50% band and collapses by 30%. That is a peak with worse neighbours on both
sides — a fitted number, not a mechanism. Contrast `vol_target`, whose monotone
25/20/15/12 response is why that lever is trusted.

Its sub-era record also fails on magnitude even while passing on count: 4/4 eras
"better", but by 0.1 / 0.6 / 0.8pts and then 7.6pts in 2020-2026. One episode
wearing a consistency badge. And on the live book it drops MU mid-memory-cycle
for a slower-growing competitor in the same industry.

**Lesson recorded:** "better in most sub-eras" is too weak a bar on its own.
Check the MAGNITUDE per era — an effect concentrated in one period is one
observation, however many buckets it is sliced into.

---

## v4 — monthly calendar rebalance · 2026-07-30

| | CAGR | Sharpe | Sortino | maxDD |
|---|---|---|---|---|
| **v4** | 17.48% | 0.900 | 1.283 | −45.69% |
| v3 *(as reported)* | 18.29% | 0.934 | 1.333 | −30.76% |

**Changed:** `rebal_months=1`, `hold` 42 → 21. Books are now dated the **first
trading day of every month** instead of falling on a 42-day stride from an
arbitrary anchor. A product decision — a book published to subscribers has to
arrive on a date they can plan around.

**The part that matters.** v3's numbers were not real. Its rebalance grid
started from whatever day sat 400 calendar days into the sample — a value that
fell out of a lookback buffer, was never chosen, and was therefore never swept.
Sliding that anchor a week at a time, holding everything else fixed:

| phase | CAGR | Sharpe | maxDD | worst drawdown |
|---|---|---|---|---|
| **0 (shipped as v3)** | **18.29%** | **0.934** | **−30.76%** | 2021-08 → **2022-01** |
| 7 | 16.95% | 0.874 | −43.62% | 2021-08 → 2023-10 |
| 14 | 17.08% | 0.884 | −43.21% | 2021-02 → 2023-10 |
| 21 | 15.79% | 0.826 | −44.79% | 2021-08 → 2023-10 |
| 28 | 15.68% | 0.824 | −47.79% | 2021-02 → 2023-10 |
| 35 | 16.33% | 0.850 | −48.15% | 2021-06 → 2023-10 |

v3 was the **best of six on every axis**, median drawdown −44.21%, spread 17.4
points. Every other phase stayed underwater until October 2023; the shipped one
recovered in January 2022 because it happened to exit the November-2021 momentum
peak on a lucky week. So **v4 is not a degradation from v3 — v3 was never real.**
Against the honest median of the v3 clock (~16.6% / ~0.87 / −44%), v4 is a wash.

The grid is now anchored to the calendar, so this class of luck is structurally
unavailable. See RESEARCH_RULES #8 and `grid_phase_test.py`.

---

## v3 — volatility targeting · 2026-07-29

Added `vol_target=0.25`, `vol_lookback=21`, `vol_cap=1.0`; `continuous_regime`
turned on. Reported 18.29% / 0.934 / −30.76% — **see v4: these were measured on
one lucky grid phase and overstate the model.** The *direction* of both levers
survived re-measurement; the magnitudes did not.

Re-measured on the v4 grid, volatility targeting remains the dominant exposure
lever and its response is monotone, which is what distinguishes a risk dial from
a fitted parameter:

| vol target | CAGR | Sharpe | maxDD | time >20% underwater |
|---|---|---|---|---|
| off | 19.12% | 0.823 | −54.88% | 29.0% |
| 25% *(shipped)* | 17.48% | 0.900 | −45.69% | 11.0% |
| 20% | 16.17% | 0.925 | −39.32% | 7.2% |
| 15% | 13.96% | 0.961 | −28.71% | 2.8% |
| 12% | 12.30% | 1.005 | −22.80% | 0.5% |

Also measured on the v4 grid: the 200-day-MA regime gate is now nearly inert —
going to *full cash* below the line moves the drawdown only −45.7% → −44.4%,
because the worst episode happened with the market above its average. That is
precisely the blind spot volatility targeting exists to cover.

---

## v2 — 12-1 momentum replaces acceleration · 2026-07-27

Acceleration was **falsified** on survivorship-free Sharadar data: +0.27% per
rebalance (t=0.4) in-sample, −6.4%/yr against the S&P out of sample, −52.8%
drawdown, dead in every size, era, sector and regime bucket tested. Its apparent
+7%/yr excess came from a panel whose 111 delisted names had all died in 2024 or
later — the backtest could not hold a stock through its death before then.

Replaced by 12-1 momentum (Jegadeesh–Titman). Same change added `sector_cap=2`
(the live book had been 9/10 Technology) and raised `mcap_floor` to $10B;
`growth_mix` was retired for ranking negatively (−1.61%/reb, t=−2.5).

---

## v1 — acceleration, pre-Sharadar · through 2026-07-26

3-month minus prior-3-month return, $2B floor, 0.50 correlation cap, 75%
revenue-growth pool, 42-day clock in two staggered sleeves. Ran on
survivorship-**biased** history. Every number it produced is void; it is
recorded here only so old screenshots can be identified.

---

## How to read this file

These are **backtests**. The rules were chosen knowing how each period turned
out, and v3 and v4 are a worked example of how much that can flatter a result
without anyone intending it. The only out-of-sample evidence this model has is
the forward record on the Edge Tracker page, which begins at each version's
ship date and is deliberately short.

When a version changes, check that the numbers quoted in `templates/model.html`,
`static/js/edge.js`, `edge_tracker_lib.MODEL_STATUS` and `README.md` moved with
it. On 2026-07-30 an audit found five stale figures on the live page that had
been correct for v3 and wrong since.
