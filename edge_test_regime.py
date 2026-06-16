"""edge_test_regime.py -- validate the 200-day-MA regime switch overlay.

Question (Jared's prototype): does raising CASH when the S&P is below its 200dMA
cut drawdowns by ~a third without wrecking returns? We test, over 1Y/2Y/5Y/MAX:
  1. baseline (always invested) vs full-cash regime switch
  2. PARTIAL de-risking: scale basket to 50% / 25% exposure below the line
  3. robustness: a buffer (band below the MA, and a 1-period reclaim-confirm) to
     avoid whipsaw near the line
  4. whipsaw frequency (switches/year) -- trading-cost concern

Does NOT modify edge_lib.py or qmodel/.
"""
from __future__ import annotations
import numpy as np
import edge_lib as E

# ---------------------------------------------------------------- setup
pan = E.load_edge_panel()
holds = [E.select_topN(df) for df in pan.panels]

# basket per-period returns with NO overlay -- we apply our own regime masks to this
basket = E.period_returns(pan, holds, regime_cash=False)
T = pan.T
PPY = E.PPY
RF = E.RF_PER
on = pan.ma200_on.astype(bool)          # True = S&P above 200dMA at that rebalance

WINDOWS = (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None))


def blended(mask_invested, expo=0.0):
    """Return per-period series. Where mask_invested is True -> full basket.
    Where False -> blend: expo*basket + (1-expo)*cash. expo=0 => full cash."""
    r = basket.copy()
    derisk = ~mask_invested
    r[derisk] = expo * basket[derisk] + (1.0 - expo) * RF
    return r


def windrow(label, rets):
    cells = []
    for _, k in WINDOWS:
        kk = T if k is None else min(k, T)
        c, dd, sh = E.perf(rets[-kk:])
        cells.append((c, sh, dd))
    return label, cells


def print_table(rows, title):
    print(f"\n{title}")
    hdr = f"  {'spec':<26}"
    for wname, _ in WINDOWS:
        hdr += f"{wname+' CAGR':>11}{'Sh':>6}{'DD':>7}"
    print(hdr)
    for label, cells in rows:
        line = f"  {label:<26}"
        for c, sh, dd in cells:
            line += f"{c*100:9.1f}%{sh:6.2f}{dd*100:6.0f}%"
        print(line)


def maxdd(rets, k=None):
    kk = T if k is None else min(k, T)
    return E.perf(rets[-kk:])[1]


# ---------------------------------------------------------------- 1. baseline vs full cash
base = basket.copy()
full = blended(on, expo=0.0)            # 0% exposure below the line (full cash)

rows1 = [windrow("baseline (always in)", base),
         windrow("full regime cash", full)]

# ---------------------------------------------------------------- 2. partial de-risking
partials = {}
for expo in (0.25, 0.50, 0.75):
    partials[expo] = blended(on, expo=expo)

rows2 = rows1 + [windrow(f"partial {int(e*100)}% exposure", partials[e])
                 for e in (0.75, 0.50, 0.25)]

# ---------------------------------------------------------------- 3. buffer / whipsaw masks
# We need a daily-ish proxy for "how far below the MA" -- but pan only gives us the
# boolean per rebalance. So the band test uses spxf magnitude as a proxy is wrong;
# instead build buffer rules from the boolean signal itself + a reclaim-confirm.

# (a) reclaim-confirm: stay in cash until the S&P has been back above the MA for
#     >=1 full rebalance period (i.e. require on[i] AND on[i-1]) before re-investing.
confirm = on.copy()
invested_confirm = np.zeros(T, bool)
state = True                            # start invested (above line at t0 typically)
for i in range(T):
    if on[i]:
        # only flip back to invested if previous period was also "on" (confirmed)
        if i == 0 or on[i - 1]:
            state = True
        # else: stay in whatever state we were (don't re-enter on a 1-bar blip)
    else:
        state = False
    invested_confirm[i] = state
confirm_full = blended(invested_confirm, expo=0.0)

# (b) exit-confirm (symmetric hysteresis): only go to cash after 2 consecutive
#     below-the-line periods, and only re-enter after the MA is reclaimed.
invested_hyst = np.zeros(T, bool)
state = True
for i in range(T):
    if state:                           # currently invested -> need 2 below to exit
        if not on[i] and (i == 0 or not on[i - 1]):
            state = False
    else:                               # currently cash -> reclaim MA to re-enter
        if on[i]:
            state = True
    invested_hyst[i] = state
hyst_full = blended(invested_hyst, expo=0.0)

rows3 = [windrow("full regime cash", full),
         windrow("reclaim-confirm (entry)", confirm_full),
         windrow("hysteresis (2-bar exit)", hyst_full)]

# ---------------------------------------------------------------- 4. whipsaw frequency
def switches(invested_mask):
    return int(np.sum(invested_mask[1:] != invested_mask[:-1]))

years = T / PPY
raw_switches = switches(on)
confirm_switches = switches(invested_confirm)
hyst_switches = switches(invested_hyst)

# ---------------------------------------------------------------- print
print("=" * 96)
print(f"Panel: {T} rebalances ({years:.1f}y), {pan.bdates[0].date()} -> {pan.bdates[-1].date()}; "
      f"S&P above 200dMA {on.mean()*100:.0f}% of periods; below {(~on).mean()*100:.0f}%")
print("=" * 96)

print_table(rows1, "1) BASELINE vs FULL REGIME CASH")
print_table(rows2, "2) PARTIAL DE-RISKING (exposure below the 200dMA)")
print_table(rows3, "3) ROBUSTNESS: buffer / hysteresis (all 0% exposure below)")

# drawdown reduction summary (MAX and 5Y, the windows where crashes live)
print("\n  Drawdown reduction vs baseline:")
for wname, k in WINDOWS:
    b = maxdd(base, k); f = maxdd(full, k)
    red = (1 - f / b) * 100 if b < 0 else 0.0
    print(f"    {wname:<5} baseline {b*100:6.0f}%  -> full cash {f*100:6.0f}%   "
          f"({red:+.0f}% reduction in maxDD)")

print("\n  Best-partial drawdown vs return (MAX window):")
for tag, r in [("baseline", base), ("75% expo", partials[0.75]),
               ("50% expo", partials[0.50]), ("25% expo", partials[0.25]),
               ("full cash", full)]:
    c, dd, sh = E.perf(r)
    print(f"    {tag:<10} CAGR {c*100:6.1f}%  Sharpe {sh:5.2f}  maxDD {dd*100:6.0f}%")

print("\n  Whipsaw / switch frequency (each switch ~= one full basket turnover):")
print(f"    raw 200dMA rule       : {raw_switches} switches over {years:.1f}y "
      f"= {raw_switches/years:.2f}/yr")
print(f"    reclaim-confirm entry : {confirm_switches} switches "
      f"= {confirm_switches/years:.2f}/yr")
print(f"    2-bar hysteresis      : {hyst_switches} switches "
      f"= {hyst_switches/years:.2f}/yr")

# days/periods spent in cash
print(f"\n    periods in cash -- raw: {int((~on).sum())}, "
      f"confirm: {int((~invested_confirm).sum())}, "
      f"hyst: {int((~invested_hyst).sum())} (of {T})")
