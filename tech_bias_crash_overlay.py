"""Crash-protection (vol-targeting) overlay -- clean HEAD-TO-HEAD vs the current model.

Recommended overlay (from prior research, tech_bias_volscale_daily.py):
  Barroso-Santa-Clara risk-managed momentum with a DAILY EWMA(lambda=0.94) vol
  forecast, de-risk-only (lev_cap=1.0), target_vol = the strategy's full-sample
  realized vol (~27%/yr). Exposure today is scaled by yesterday's forecast (no
  look-ahead); the un-invested fraction earns RF daily.

This script nails that overlay down cleanly and compares it head-to-head with the
CURRENT (un-scaled) model:

  variant | CAGR | Sharpe | maxDD | 2008 | 2009 | 2020 | 2022 | gross alpha %/yr (t)

Variants:
  - CURRENT model            : baseline, model_returns(...)['ret'] (no scaling)
  - daily EWMA.94 cap1.0     : recommended de-risk-only overlay
  - daily EWMA.94 cap1.5     : aggressive alternative (allows up to 1.5x leverage)

Both overlay variants are built on the DAILY strategy series, scaled daily, then
aggregated to the 21-day rebalance grid (compound daily returns within each
HOLD-day window) so that CAGR / Sharpe / maxDD / alpha are apples-to-apples with
the period-grid baseline. (Raw daily-grid Sharpe is artificially depressed by
daily autocorrelation; we report PERIOD-GRID Sharpe throughout.)

Alpha is via the full Carhart (MKT/SMB/HML/UMD) + in-universe sector-tilt controls,
Newey-West t-stats.

Does NOT modify tech_bias_lib.py or any live-model file.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import tech_bias_lib as L
from qmodel.equations import default_params

RF_ANNUAL = 0.04
RF_DAILY = RF_ANNUAL / 252.0
TD = 252.0


# ---------------------------------------------------------------------------
# daily EWMA vol forecast + Barroso-Santa-Clara scaling (no look-ahead)
# ---------------------------------------------------------------------------
def ewma_forecast_vol(ds: pd.Series, lam: float = 0.94) -> pd.Series:
    """Annualized RiskMetrics EWMA daily-vol forecast:
    var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2  (mean-zero convention)."""
    var = ds.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var) * np.sqrt(TD)


def vol_scaled_daily(ds: pd.Series, target_vol: float, lam: float = 0.94,
                     lev_cap: float = 1.0) -> pd.Series:
    """Scale today's daily return by min(target/forecast_{t-1}, lev_cap). The
    un-invested fraction (1-lev) earns RF daily. Uses the PREVIOUS day's
    forecast -> no look-ahead."""
    fc = ewma_forecast_vol(ds, lam=lam)
    lev = (target_vol / fc).clip(upper=lev_cap).shift(1)
    scaled = lev * ds + (1.0 - lev) * RF_DAILY
    return scaled.dropna()


# ---------------------------------------------------------------------------
# aggregate a daily series to the panel's HOLD-day rebalance grid (1:1 w/ bdates
# and the factor arrays), by compounding daily returns within each window.
# ---------------------------------------------------------------------------
def to_period_grid(s_daily: pd.Series, pan) -> np.ndarray:
    cal = s_daily.index
    out = np.full(pan.T, np.nan)
    for i, d in enumerate(pan.bdates):
        loc = int(cal.searchsorted(d, side="right"))
        win = cal[loc: loc + L.HOLD]
        if len(win) == 0:
            continue
        out[i] = float((1 + s_daily.loc[win]).prod() - 1)
    return out


# ---------------------------------------------------------------------------
# calendar-year return from a period-grid array aligned to pan.bdates
# ---------------------------------------------------------------------------
def cal_year_period(arr: np.ndarray, pan, yr: int) -> float:
    s = pd.Series(np.nan_to_num(arr), index=pan.bdates)
    sub = s[s.index.year == yr]
    if len(sub) == 0:
        return np.nan
    return float((1 + sub).prod() - 1)


# ---------------------------------------------------------------------------
# table formatting
# ---------------------------------------------------------------------------
COLS = ["variant", "CAGR", "Sharpe", "maxDD", "2008", "2009", "2020", "2022",
        "alpha %/yr (t)"]


def _cell(r, c):
    v = r[c]
    if c == "variant" or c == "alpha %/yr (t)":
        return str(v)
    if c == "Sharpe":
        return f"{v:.2f}"
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v*100:+.1f}%"


def fmt(rows):
    w = {c: max(len(c), *(len(_cell(r, c)) for r in rows)) for c in COLS}
    out = ["  ".join(c.ljust(w[c]) for c in COLS),
           "  ".join("-" * w[c] for c in COLS)]
    for r in rows:
        out.append("  ".join(_cell(r, c).ljust(w[c]) for c in COLS))
    return "\n".join(out)


def make_row(name, period_ret, pan, regs):
    """period_ret: per-period (21d) return array aligned to pan.bdates."""
    mask = ~np.isnan(period_ret)
    cagr, dd, shp = L.perf(period_ret[mask])
    a_ann, a_t, _ = L.attribution(name, period_ret, regs, mask=mask, verbose=False)
    return {"variant": name, "CAGR": cagr, "Sharpe": shp, "maxDD": dd,
            "2008": cal_year_period(period_ret, pan, 2008),
            "2009": cal_year_period(period_ret, pan, 2009),
            "2020": cal_year_period(period_ret, pan, 2020),
            "2022": cal_year_period(period_ret, pan, 2022),
            "alpha %/yr (t)": f"{a_ann*100:+.1f}% (t={a_t:+.2f})"}


# ---------------------------------------------------------------------------
def main():
    pan = L.load_panel()
    params = default_params()

    # ---- baseline: current (un-scaled) model on the period grid ----
    base = L.model_returns(pan, params, vol_scale=False)["ret"]

    # ---- daily strategy series + target vol (full-sample realized) ----
    ds = L.daily_strategy_returns(pan, params).sort_index()
    target_vol = float(ds.std(ddof=0) * np.sqrt(TD))
    print(f"daily strategy series : {len(ds)} days  "
          f"{ds.index[0].date()} -> {ds.index[-1].date()}")
    print(f"target_vol (full-sample realized) = {target_vol*100:.1f}%/yr\n")

    # ---- build the two overlay variants on the daily grid, aggregate to period grid ----
    ds_cap10 = vol_scaled_daily(ds, target_vol, lam=0.94, lev_cap=1.0)
    ds_cap15 = vol_scaled_daily(ds, target_vol, lam=0.94, lev_cap=1.5)

    pg_cap10 = to_period_grid(ds_cap10, pan)
    pg_cap15 = to_period_grid(ds_cap15, pan)

    # ---- regressors: full Carhart + in-universe sector tilts ----
    regs = {**L.carhart_factors(pan), **L.sector_tilts(pan)}

    rows = [
        make_row("CURRENT model (baseline)", base, pan, regs),
        make_row("daily EWMA.94 cap1.0", pg_cap10, pan, regs),
        make_row("daily EWMA.94 cap1.5", pg_cap15, pan, regs),
    ]

    print("=== HEAD-TO-HEAD (all on the 21-day period grid; alpha = Carhart + sector, NW) ===")
    print(fmt(rows))
    print()

    # ---- detailed attribution dump for the recommended overlay ----
    print("=== ATTRIBUTION DETAIL ===")
    print("CURRENT model (baseline):")
    L.attribution("baseline", base, regs)
    print("daily EWMA.94 cap1.0 (recommended):")
    m10 = ~np.isnan(pg_cap10); L.attribution("EWMA cap1.0", pg_cap10, regs, mask=m10)
    print("daily EWMA.94 cap1.5 (aggressive):")
    m15 = ~np.isnan(pg_cap15); L.attribution("EWMA cap1.5", pg_cap15, regs, mask=m15)


if __name__ == "__main__":
    main()
