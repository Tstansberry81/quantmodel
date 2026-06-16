"""The Edge -- combined backtest of the recommended spec (capstone).

Stacks the six agents' winning pieces into one model and reports an honest
number over 1Y / 2Y / 5Y / MAX:
  * Signal      : acceleration (3-mo minus prior-3-mo momentum)   [signal agent]
  * Clock       : ~60-day hold (hold=42), 20-name basket          [signal agent]
  * Liquidity   : >= $2B market-cap floor (tradeable proxy)       [liquidity agent]
  * Correlation : point-in-time cap at 0.50                       [corr-cap agent]
  * Regime      : 25% exposure (75% cash) below the S&P 200dMA    [regime agent]
  * Volume      : used as the liquidity gate, NOT a ranker        [volume agent]
Then a survivorship haircut (~-3%/yr) per the survivorship agent.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import edge_lib as E

HOLD, NBASKET, MCAP_FLOOR, CAP, REG_EXPO = 42, 20, 2e9, 0.50, 0.25
W = {"accel": 1.0}
HAIRCUT = 0.03           # survivorship discount to apply to CAGR

print("Building 60-day-clock panel ...")
pan = E.load_edge_panel(hold=HOLD)
print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n")


def liq(df):
    return df[df["pit_mcap"] >= MCAP_FLOOR]


def returns(select, floor=True, regime=False):
    holds = [select(liq(df) if floor else df, pan.bdates[i]) for i, df in enumerate(pan.panels)]
    r = E.period_returns(pan, holds)
    if regime:
        r = np.where(pan.ma200_on, r, REG_EXPO * r + (1 - REG_EXPO) * E.RF_PER)
    return r


pick_plain = lambda df, d: E.select_topN(df, n=NBASKET, weights=W)
pick_cap = lambda df, d: E.corr_cap_select(df, asof=d, n=NBASKET, weights=W, cap=CAP)

# how much does the $2B floor remove?
med_all = np.median([df["pit_mcap"].median() for df in pan.panels])
med_flo = np.median([liq(df)["pit_mcap"].median() for df in pan.panels])
drop = np.mean([1 - len(liq(df)) / len(df) for df in pan.panels])
print(f"Liquidity floor >=$2B drops {drop*100:.0f}% of names per period "
      f"(universe median mcap ${med_all/1e9:.1f}B -> floored ${med_flo/1e9:.1f}B)\n")

print("=" * 70)
print("BUILD-UP: each row adds one piece (CAGR / Sharpe / maxDD per window)")
print("=" * 70)
E.eval_windows(pan, returns(pick_plain, floor=False), "1) accel signal, 60d, n=20 (no overlays, full universe)")
E.eval_windows(pan, returns(pick_plain, floor=True), "2) + liquidity floor >=$2B")
E.eval_windows(pan, returns(pick_cap, floor=True), "3) + correlation cap 0.50")
full = returns(pick_cap, floor=True, regime=True)
E.eval_windows(pan, full, "4) + regime switch (FULL EDGE)")

print("\n" + "=" * 70)
print("HONEST HEADLINE -- Full Edge, with survivorship haircut")
print("=" * 70)
for wname, k in (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None)):
    kk = pan.T if k is None else min(k, pan.T)
    c, dd, sh = E.perf(full[-kk:])
    cs, _, ss = E.perf(pan.spxf[-kk:])
    print(f"  {wname:<4} gross CAGR {c*100:5.1f}%  ->  net of ~3%/yr survivorship "
          f"{(c-HAIRCUT)*100:5.1f}%   Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%   "
          f"(S&P {cs*100:.0f}%/{ss:.2f})")
print("\nNote: hold=42 ~ 60-day clock. Liquidity floor uses market cap as a full-history")
print("proxy; the volume agent's $10-25M/day dollar-volume gate (2018+) is the live gate.")
