"""Model vs. BUYABLE alternatives -- the skeptical-allocator test.

WHY THIS SCRIPT EXISTS
----------------------
tech_bias_external.py already showed the model survives REAL Ken-French Carhart
factors (alpha ~ +9.5%/yr, t=2.89; +10.8% adding SPDR sectors). But Carhart
factors are long-short academic portfolios you cannot literally buy. The
practical question a skeptical allocator asks is sharper:

  "Could I get most of this just by buying a cheap momentum/factor ETF?
   How much does the model add OVER the best BUYABLE alternative?"

This script answers that by:
  1. Pulling BUYABLE benchmarks via yfinance (monthly total returns): MTUM, QUAL,
     SPMO, SPY, QQQ, IWM, plus ^SP500TR; and the REAL Ken-French monthly Carhart
     factors (download/parse reused from tech_bias_external.py).
  2. A head-to-head table (CAGR / Sharpe / maxDD) over each pair's COMMON window.
  3. The key number: regress the model's monthly EXCESS return on the best
     buyable momentum alternative (MTUM excess), then a Mkt+UMD 2-factor, then
     full Carhart -- the alpha left AFTER you could just buy the ETF.
  4. A plain-English verdict.

Does NOT modify tech_bias_lib.py or any live-model file.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import tech_bias_lib
from tech_bias_lib import ols_nw, RF_PER
from qmodel.equations import default_params

# Reuse the REAL French Carhart download/parse from the existing external script.
from tech_bias_external import load_french_monthly, load_etf_proxy_monthly

PPY_M = 12.0                      # we work in calendar months here
RF_M = 0.04 / 12.0               # config risk-free, monthly (matches RF_PER*PPY/12)

BENCH_TICKERS = {
    "MTUM":   "iShares MSCI USA Momentum (since ~2013)",
    "SPMO":   "Invesco S&P 500 Momentum",
    "QUAL":   "iShares MSCI USA Quality",
    "QQQ":    "Invesco QQQ (Nasdaq-100)",
    "SPY":    "SPDR S&P 500",
    "IWM":    "iShares Russell 2000 (small cap)",
    "^SP500TR": "S&P 500 Total Return index",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def monthly_total_returns(tickers: list[str], start="2003-01-01") -> pd.DataFrame:
    """Monthly total returns from yfinance auto-adjusted closes.
    Months with no actual daily observation are masked to NaN (so dropna()
    trims each series to its true live history -- avoids fabricating 0% months
    before an ETF's inception, e.g. MTUM/SPMO/QUAL)."""
    import yfinance as yf
    px = yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    dr = px.pct_change()
    r = (1 + dr).resample("ME").prod() - 1
    valid = dr.notna().resample("ME").sum() > 0
    r = r.where(valid)
    r.index = r.index.to_period("M")
    return r


def model_monthly() -> pd.Series:
    """Strategy daily returns compounded to calendar months (PeriodIndex)."""
    pan = tech_bias_lib.load_panel()
    daily = tech_bias_lib.daily_strategy_returns(pan, default_params())
    m = (1.0 + daily).resample("ME").prod() - 1.0
    m.index = m.index.to_period("M")
    return m[m != 0.0].dropna()        # drop empty leading/trailing months


# ---------------------------------------------------------------------------
# stats helpers (calendar-monthly)
# ---------------------------------------------------------------------------
def cagr(r: np.ndarray) -> float:
    eq = np.prod(1 + r)
    return eq ** (PPY_M / len(r)) - 1 if eq > 0 else -1.0


def sharpe(r: np.ndarray, rf_m=RF_M) -> float:
    ex = r - rf_m
    sd = np.std(ex, ddof=1)
    return float(np.sqrt(PPY_M) * np.mean(ex) / sd) if sd > 0 else np.nan


def maxdd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def ann_alpha(a_m: float) -> float:
    return (1 + a_m) ** PPY_M - 1


def regress(y, regressors: dict, L: int = 6):
    names = ["alpha"] + list(regressors)
    X = np.column_stack([np.ones(len(y))] + [regressors[k] for k in regressors])
    beta, se, t = ols_nw(y, X, L=L)
    resid = y - X @ beta
    ss_res = float(resid @ resid); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, names, r2


def tag_t(t):
    return "** survives (t>2)" if abs(t) > 2 else "NOT sig (t<1.65)" if abs(t) < 1.65 else "~ marginal"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("MODEL vs. BUYABLE ALTERNATIVES -- 'could I just buy a momentum/factor ETF?'")
    print("=" * 80)

    # --- factors (REAL French, ETF proxy fallback) ---
    try:
        fac, FAC_SRC = load_french_monthly()
    except Exception as e:
        print(f"  [French download failed: {repr(e)[:70]}] -> ETF proxy")
        fac, FAC_SRC = load_etf_proxy_monthly()

    # --- buyable benchmarks ---
    print("\nloading buyable ETFs/indices (yfinance) ...")
    bench = monthly_total_returns(list(BENCH_TICKERS))

    # --- model monthly ---
    print("loading rebalance panel + building model monthly returns (~1 min) ...")
    model = model_monthly()

    print("\n(a) DATA SOURCES")
    print(f"    factors : {FAC_SRC}")
    print(f"              coverage {fac.index.min()} .. {fac.index.max()} ({len(fac)} mo)")
    print(f"    buyable : yfinance monthly total returns -- " +
          ", ".join(f"{k}" for k in BENCH_TICKERS))
    for k, desc in BENCH_TICKERS.items():
        if k in bench.columns:
            col = bench[k].dropna()
            print(f"              {k:<9} {desc:<42} {col.index.min()}..{col.index.max()} ({len(col)} mo)")
    print(f"    model   : strategy daily returns -> calendar months "
          f"{model.index.min()} .. {model.index.max()} ({len(model)} mo)")

    # =======================================================================
    # (b) HEAD-TO-HEAD PERFORMANCE TABLE (each pair's OWN overlapping window)
    # =======================================================================
    print("\n" + "-" * 80)
    print("(b) HEAD-TO-HEAD: model vs each buyable, over the pair's COMMON window")
    print("-" * 80)
    compare = ["MTUM", "SPMO", "QQQ", "SPY"]

    def row(label, r):
        return f"    {label:<26} {cagr(r)*100:+7.2f}%  {sharpe(r):+6.2f}  {maxdd(r)*100:+8.2f}%   n={len(r)}"

    print(f"    {'series':<26} {'CAGR':>8}  {'Sharpe':>6}  {'maxDD':>9}")
    for b in compare:
        if b not in bench.columns:
            continue
        bser = bench[b].dropna()
        common = model.index.intersection(bser.index).sort_values()
        if len(common) < 12:
            continue
        m = model.reindex(common).to_numpy(float)
        x = bser.reindex(common).to_numpy(float)
        win = f"{common.min()}..{common.max()}"
        print(f"\n    [vs {b}]  common window {win}")
        print(row("MODEL", m))
        print(row(f"{b} ({BENCH_TICKERS[b].split('(')[0].strip()})", x))

    # =======================================================================
    # (c) THE KEY NUMBER: alpha the model adds OVER the best buyable momentum
    # =======================================================================
    print("\n" + "-" * 80)
    print("(c) ALPHA OVER WHAT YOU COULD JUST BUY  (model EXCESS ret regressed on ...)")
    print("-" * 80)

    results = {}

    # --- (c1) single best buyable momentum: MTUM excess ---
    mtum = bench["MTUM"].dropna() if "MTUM" in bench.columns else pd.Series(dtype=float)
    common = model.index.intersection(mtum.index)
    common = common.intersection(fac.index).sort_values()
    if len(common) >= 24:
        m = model.reindex(common).to_numpy(float)
        rf = fac["RF"].reindex(common).to_numpy(float)
        y = m - rf
        mtum_ex = mtum.reindex(common).to_numpy(float) - rf
        beta, se, t, names, r2 = regress(y, {"MTUM-RF": mtum_ex})
        a = ann_alpha(beta[0])
        results["MTUM"] = (a, float(t[0]), len(common))
        print(f"\n  (c1) vs MTUM alone (the single best BUYABLE momentum ETF), n={len(common)}")
        print(f"       window {common.min()}..{common.max()}")
        print(f"       alpha = {beta[0]*100:+.3f}%/mo  ANNUALIZED {a*100:+.2f}%/yr  t={t[0]:+.2f}  {tag_t(t[0])}")
        print(f"       MTUM-RF beta = {beta[1]:+.3f}  t={t[1]:+.2f}   R^2={r2:.3f}")

    # --- (c2) Mkt + UMD two-factor (buyable market + the momentum premium) ---
    common = model.index.intersection(fac.index).sort_values()
    m = model.reindex(common).to_numpy(float)
    rf = fac["RF"].reindex(common).to_numpy(float)
    y = m - rf
    F = fac.reindex(common)
    reg2 = {"Mkt-RF": F["Mkt-RF"].to_numpy(float), "UMD": F["UMD"].to_numpy(float)}
    beta, se, t, names, r2 = regress(y, reg2)
    a = ann_alpha(beta[0]); results["Mkt+UMD"] = (a, float(t[0]), len(common))
    print(f"\n  (c2) vs Mkt-RF + UMD (market + momentum premium), n={len(common)}")
    print(f"       window {common.min()}..{common.max()}")
    print(f"       alpha = {beta[0]*100:+.3f}%/mo  ANNUALIZED {a*100:+.2f}%/yr  t={t[0]:+.2f}  {tag_t(t[0])}")
    for i, nm in enumerate(names[1:], 1):
        print(f"       {nm:<8} loading={beta[i]:+.3f}  t={t[i]:+.2f}")
    print(f"       R^2={r2:.3f}")

    # --- (c3) full Carhart (4-factor) ---
    reg4 = dict(reg2); reg4["SMB"] = F["SMB"].to_numpy(float); reg4["HML"] = F["HML"].to_numpy(float)
    reg4 = {"Mkt-RF": F["Mkt-RF"].to_numpy(float), "SMB": F["SMB"].to_numpy(float),
            "HML": F["HML"].to_numpy(float), "UMD": F["UMD"].to_numpy(float)}
    beta, se, t, names, r2 = regress(y, reg4)
    a = ann_alpha(beta[0]); results["Carhart"] = (a, float(t[0]), len(common))
    print(f"\n  (c3) vs FULL CARHART (Mkt-RF + SMB + HML + UMD), n={len(common)}")
    print(f"       window {common.min()}..{common.max()}")
    print(f"       alpha = {beta[0]*100:+.3f}%/mo  ANNUALIZED {a*100:+.2f}%/yr  t={t[0]:+.2f}  {tag_t(t[0])}")
    for i, nm in enumerate(names[1:], 1):
        print(f"       {nm:<8} loading={beta[i]:+.3f}  t={t[i]:+.2f}")
    print(f"       R^2={r2:.3f}")

    # =======================================================================
    # (d) VERDICT
    # =======================================================================
    print("\n" + "=" * 80)
    print("(d) VERDICT -- is the model worth running vs just buying a momentum/factor ETF?")
    print("=" * 80)
    for k in ("MTUM", "Mkt+UMD", "Carhart"):
        if k in results:
            a, tt, n = results[k]
            print(f"    alpha over {k:<9}: {a*100:+6.2f}%/yr   t={tt:+.2f}   (n={n})   {tag_t(tt)}")
    print()
    print("    Read the head-to-head table (b) for raw CAGR/Sharpe/maxDD vs the ETFs,")
    print("    and (c) for the alpha that survives AFTER replicating with buyable")
    print("    momentum/factor exposure. Verdict text written by the researcher.")


if __name__ == "__main__":
    main()
