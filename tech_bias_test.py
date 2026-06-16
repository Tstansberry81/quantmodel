"""Tech-bias / alpha-attribution diagnostic  (coworker's request).

Question: is the strategy's S&P/Nasdaq outperformance genuine stock-PICKING
skill, or just (a) a leveraged bet on the market, (b) a tech tilt, or (c) the
well-known momentum premium you could buy passively?

The honest test is a FACTOR REGRESSION. We explain the strategy's monthly
returns with a set of things you could get without skill -- the market, a size
bet, a value bet, a momentum bet, and sector tilts -- and see what's LEFT OVER
(the "alpha" / intercept). If a real, statistically-significant chunk is left
after stripping all of those out, that's genuine selection skill. If it
collapses to ~zero, the outperformance was just factor exposure.

Sections:
  1. Signal hygiene (12-1 skip-month momentum).
  2. What the book actually holds (sector mix + market beta).
  3. THE TEST -- step up the controls: market -> +sectors -> +size/value/momentum.
  4. Is the edge stable, or a lucky window? (first half / second half / ex-2020)
  5. Sector-neutral vs cross-sectional scoring.
  6. Risk-managed (vol-scaled) variant.
  7. Crisis years: 2008, 2009, 2020, 2022.

Standalone research script; reads cached artifacts via tech_bias_lib (shared
with the other research scripts). Does not touch the live model.
"""
import warnings; warnings.filterwarnings("ignore")
import copy
import numpy as np
import tech_bias_lib as L
from qmodel.equations import default_params

print("Building rebalance panel (one-time, ~1 min) ...")
pan = L.load_panel()
print(f"Panel: {pan.T} monthly rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n")

base = default_params()                                  # current live default
xsec = copy.deepcopy(base); xsec["settings"]["sector_neutral"] = False

mb = L.model_returns(pan, base)                          # default config
mx = L.model_returns(pan, xsec)                          # cross-sectional scoring
m_base = mb["ret"]

# factor + sector-tilt regressors built once
F = L.carhart_factors(pan)                               # MKT_RF, SMB, HML, UMD
TILTS = L.sector_tilts(pan)

print("=" * 80)
print("1. SIGNAL HYGIENE -- 12-1 skip-month momentum")
print("=" * 80)
print("  momentum = price 1 month ago / price 12 months ago - 1 (skips last ~month)")
print("  -> Confirmed 12-1, not 12-0. Short-term reversal is a separate factor.\n")

print("=" * 80)
print("2. WHAT THE BOOK ACTUALLY HOLDS (default config)")
print("=" * 80)
c, dd, sh = L.perf(m_base)
print(f"  Full sample: CAGR {c*100:.1f}%   Sharpe {sh:.2f}   worst drawdown {dd*100:.0f}%")
print(f"  Avg weight in Tech + Communications : {np.mean(mb['tech_w'])*100:.0f}%   "
      f"(S&P is ~30%+; so NOT tech-heavy)")
print(f"  Avg market beta of the book         : {np.nanmean(mb['beta']):.2f}   "
      f"(>1 = a leveraged-index bet)\n")

print("=" * 80)
print("3. THE TEST -- strip out factor exposures, see what alpha is LEFT")
print("=" * 80)
print("  Each row adds more 'free' exposures to control for. Watch the alpha.\n")
L.attribution("vs market only", m_base, {"MKT": F["MKT_RF"]})
L.attribution("vs market + sector tilts", m_base, {"MKT": F["MKT_RF"], **TILTS})
L.attribution("vs market + size + value + MOMENTUM (Carhart)", m_base,
              {"MKT": F["MKT_RF"], "SMB": F["SMB"], "HML": F["HML"], "UMD": F["UMD"]})
L.attribution("vs EVERYTHING (Carhart + sector tilts)", m_base,
              {"MKT": F["MKT_RF"], "SMB": F["SMB"], "HML": F["HML"], "UMD": F["UMD"], **TILTS})

print()
print("=" * 80)
print("4. IS THE EDGE STABLE, OR A LUCKY WINDOW? (Carhart + sectors)")
print("=" * 80)
full = {"MKT": F["MKT_RF"], "SMB": F["SMB"], "HML": F["HML"], "UMD": F["UMD"], **TILTS}
half = pan.T // 2
yr = pan.bdates.year
L.attribution(f"first half  {pan.bdates[0].date()}..{pan.bdates[half-1].date()}",
              m_base, full, mask=np.arange(pan.T) < half)
L.attribution(f"second half {pan.bdates[half].date()}..{pan.bdates[-1].date()}",
              m_base, full, mask=np.arange(pan.T) >= half)
L.attribution("excluding the +144% year 2020", m_base, full, mask=(yr != 2020))

print()
print("=" * 80)
print("5. SECTOR-NEUTRAL vs CROSS-SECTIONAL scoring")
print("=" * 80)
for nm, r, tw in [("cross-sectional (off)", mx["ret"], mx["tech_w"]),
                  ("sector-neutral  (on) ", mb["ret"], mb["tech_w"])]:
    c, dd, sh = L.perf(r)
    print(f"  {nm}: CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%  "
          f"tech-w {np.mean(tw)*100:.0f}%")

print()
print("=" * 80)
print("6. RISK-MANAGED -- vol-scaling (Barroso-Santa-Clara, monthly proxy)")
print("=" * 80)
mvs = L.model_returns(pan, base, vol_scale=True, lev_cap=1.0)
mvl = L.model_returns(pan, base, vol_scale=True, lev_cap=1.5)
for nm, r in [("raw default        ", mb["ret"]),
              ("vol-scaled (cap 1.0)", mvs["ret"]),
              ("vol-scaled (cap 1.5)", mvl["ret"])]:
    c, dd, sh = L.perf(r)
    print(f"  {nm}: CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%")
print("  (a true daily-vol version reacts faster -- handed to a separate agent)")

print()
print("=" * 80)
print("7. CRISIS YEARS (model vs S&P vs Nasdaq, calendar-year returns)")
print("=" * 80)
print("  year      model       S&P     Nasdaq    vol-scaled")
for y in [2008, 2009, 2020, 2022]:
    m = yr == y
    if m.sum() == 0:
        continue
    def cum(a): return np.prod(1 + np.nan_to_num(a[m])) - 1
    print(f"  {y}   {cum(mb['ret'])*100:+7.1f}%  {cum(pan.spxf)*100:+7.1f}%  "
          f"{cum(pan.ndxf)*100:+7.1f}%   {cum(mvs['ret'])*100:+7.1f}%")
print("\nDone.")
