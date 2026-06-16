"""Risk-managed momentum (Barroso & Santa-Clara 2015) via DAILY vol-targeting.

Thesis: momentum's long-run drag comes from rare "momentum crashes" (2009
rebound, 2020 crash). Scaling exposure by target_vol / forecast_vol, where
forecast_vol reacts in DAYS (daily realized vol), cuts those crashes and lifts
Sharpe -- and should beat the library's COARSE monthly (6-period) proxy.

We:
  1. take the daily strategy series ds = daily_strategy_returns(pan, defaults),
  2. build a daily realized-vol forecast (windows 21/42/63 + EWMA lambda=0.94),
     scale today's return by min(target/forecast_{t-1}, lev_cap) -- no look-ahead;
     uninvested fraction earns RF (daily),
  3. compare raw / coarse-monthly-proxy / daily variants on CAGR, Sharpe, maxDD
     and the 2008/2009/2020/2022 calendar-year returns,
  4. aggregate the best daily series back to the 21-day rebalance grid and run
     Carhart + sector attribution to confirm alpha is preserved.

Does NOT modify tech_bias_lib or any live-model file.
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
# daily vol-scaling
# ---------------------------------------------------------------------------
def daily_forecast_vol(ds: pd.Series, window: int | None = None,
                       ewma_lambda: float | None = None) -> pd.Series:
    """Annualized daily-vol forecast. Either a trailing rolling-std `window`
    or a RiskMetrics EWMA (var_t = lambda*var_{t-1} + (1-lambda)*r_{t-1}^2)."""
    if ewma_lambda is not None:
        # pandas ewm var with the RiskMetrics recursion (mean=0 convention).
        var = ds.pow(2).ewm(alpha=1 - ewma_lambda, adjust=False).mean()
        return np.sqrt(var) * np.sqrt(TD)
    return ds.rolling(window).std(ddof=0) * np.sqrt(TD)


def vol_scaled_daily(ds: pd.Series, target_vol: float, *, window=None,
                     ewma_lambda=None, lev_cap=1.0) -> pd.Series:
    """Barroso-Santa-Clara on the daily grid. Exposure today uses the PREVIOUS
    day's forecast (no look-ahead). Uninvested fraction earns RF daily."""
    fc = daily_forecast_vol(ds, window=window, ewma_lambda=ewma_lambda)
    lev = (target_vol / fc).clip(upper=lev_cap)
    lev = lev.shift(1)                       # use yesterday's forecast for today
    scaled = lev * ds + (1.0 - lev) * RF_DAILY
    return scaled.dropna()


# ---------------------------------------------------------------------------
# performance / reporting helpers (daily grid)
# ---------------------------------------------------------------------------
def perf_daily(s: pd.Series):
    s = s.dropna()
    eq = (1 + s).cumprod()
    yrs = len(s) / TD
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = float((eq / eq.cummax() - 1).min())
    excess = s - RF_DAILY
    shp = float(np.sqrt(TD) * excess.mean() / s.std(ddof=0))
    return float(cagr), dd, shp


def cal_year(s: pd.Series, yr: int):
    sub = s[s.index.year == yr]
    if len(sub) == 0:
        return np.nan
    return float((1 + sub).prod() - 1)


def row(name: str, s: pd.Series):
    cagr, dd, shp = perf_daily(s)
    return {"variant": name, "CAGR": cagr, "Sharpe": shp, "maxDD": dd,
            "2008": cal_year(s, 2008), "2009": cal_year(s, 2009),
            "2020": cal_year(s, 2020), "2022": cal_year(s, 2022)}


def fmt(rows):
    cols = ["variant", "CAGR", "Sharpe", "maxDD", "2008", "2009", "2020", "2022"]
    w = {c: max(len(c), *(len(_cell(r, c)) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(w[c]) for c in cols)
    out = [line, "  ".join("-" * w[c] for c in cols)]
    for r in rows:
        out.append("  ".join(_cell(r, c).ljust(w[c]) for c in cols))
    return "\n".join(out)


def _cell(r, c):
    v = r[c]
    if c == "variant":
        return str(v)
    if c == "Sharpe":
        return f"{v:.2f}"
    return f"{v*100:+.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "n/a"


# ---------------------------------------------------------------------------
def main():
    pan = L.load_panel()
    params = default_params()

    # 1. raw daily strategy
    ds = L.daily_strategy_returns(pan, params)
    ds = ds.sort_index()
    print(f"daily strategy series: {len(ds)} days  "
          f"{ds.index[0].date()} -> {ds.index[-1].date()}")

    target_vol = float(ds.std(ddof=0) * np.sqrt(TD))   # full-sample realized vol
    print(f"target_vol (full-sample realized) = {target_vol*100:.1f}%/yr\n")

    # 2. coarse monthly proxy from the library (per-period grid)
    mm = L.model_returns(pan, params, vol_scale=True)["ret"]
    raw_period = L.model_returns(pan, params, vol_scale=False)["ret"]

    # period-grid perf for the two library series (PPY ~= 12)
    def perf_period_row(name, ret):
        cagr, dd, shp = L.perf(ret)
        # build a period-indexed series on bdates for calendar-year slicing
        s = pd.Series(ret, index=pan.bdates)
        yr = lambda y: float((1 + s[s.index.year == y]).prod() - 1) \
            if (s.index.year == y).any() else np.nan
        return {"variant": name, "CAGR": cagr, "Sharpe": shp, "maxDD": dd,
                "2008": yr(2008), "2009": yr(2009), "2020": yr(2020), "2022": yr(2022)}

    rows = []
    rows.append(row("raw daily strategy", ds))
    rows.append(perf_period_row("lib coarse monthly (vol_scale=True)", mm))

    # 3. daily vol-scale grid search
    configs = []
    for win in (21, 42, 63):
        for cap in (1.0, 1.5, 2.0):
            configs.append((f"daily w{win} cap{cap}", dict(window=win, lev_cap=cap)))
    for cap in (1.0, 1.5, 2.0):
        configs.append((f"daily EWMA.94 cap{cap}", dict(ewma_lambda=0.94, lev_cap=cap)))

    daily_series = {}
    grid_rows = []
    for name, kw in configs:
        s = vol_scaled_daily(ds, target_vol, **kw)
        daily_series[name] = s
        grid_rows.append(row(name, s))

    print("=== FULL DAILY VOL-SCALE GRID ===")
    print(fmt(grid_rows))
    print()

    # pick best by Sharpe among de-risk-only (cap1.0) and among all
    best_overall = max(grid_rows, key=lambda r: r["Sharpe"])
    best_cap1 = max([r for r in grid_rows if "cap1.0" in r["variant"]],
                    key=lambda r: r["Sharpe"])
    print(f"best daily (any cap) by Sharpe : {best_overall['variant']}  "
          f"Sharpe={best_overall['Sharpe']:.2f}")
    print(f"best daily (de-risk only cap1.0): {best_cap1['variant']}  "
          f"Sharpe={best_cap1['Sharpe']:.2f}\n")

    # 4. headline comparison table
    head_rows = [
        rows[0],                              # raw daily
        rows[1],                              # coarse monthly proxy
        next(r for r in grid_rows if r["variant"] == best_cap1["variant"]),
        next(r for r in grid_rows if r["variant"] == best_overall["variant"]),
    ]
    print("=== HEADLINE COMPARISON (raw/daily rows on DAILY grid; lib row on PERIOD grid) ===")
    print(fmt(head_rows))
    print("NOTE: daily-grid Sharpe of a 21d-rebalanced strategy is depressed by")
    print("daily noise vs a period-grid Sharpe -- not comparable across grids.\n")

    # --- apples-to-apples: re-state every variant's Sharpe on the PERIOD grid ---
    def to_period_grid_q(s_daily):
        cal = s_daily.index
        out = np.full(pan.T, np.nan)
        for i, d in enumerate(pan.bdates):
            loc = int(cal.searchsorted(d, side="right"))
            win = cal[loc: loc + L.HOLD]
            if len(win):
                out[i] = float((1 + s_daily.loc[win]).prod() - 1)
        return out

    pg_rows = []
    for nm, ser in (("raw daily", ds),
                    (best_cap1["variant"], daily_series[best_cap1["variant"]]),
                    (best_overall["variant"], daily_series[best_overall["variant"]])):
        pg = to_period_grid_q(ser); m = ~np.isnan(pg)
        c, dd, sh = L.perf(pg[m])
        pg_rows.append({"variant": nm + " [period grid]", "CAGR": c, "Sharpe": sh,
                        "maxDD": dd, "2008": np.nan, "2009": np.nan,
                        "2020": np.nan, "2022": np.nan})
    pg_rows.append(rows[1])  # lib coarse monthly already on period grid
    print("=== PERIOD-GRID SHARPE (comparable to lib's 0.96->0.98) ===")
    print(fmt(pg_rows))
    print()

    # 5. alpha preservation: aggregate best daily series to the 21-day grid
    #    (compound daily returns within each rebalance window) and run attribution.
    cf = L.carhart_factors(pan)
    st = L.sector_tilts(pan)
    regs = {**cf, **st}

    def to_period_grid(s_daily: pd.Series) -> np.ndarray:
        """Compound a daily series into the panel's HOLD-day rebalance windows,
        aligned 1:1 with pan.bdates / the factor arrays."""
        cal = s_daily.index
        out = np.full(pan.T, np.nan)
        for i, d in enumerate(pan.bdates):
            loc = int(cal.searchsorted(d, side="right"))
            win = cal[loc: loc + L.HOLD]
            if len(win) == 0:
                continue
            out[i] = float((1 + s_daily.loc[win]).prod() - 1)
        return out

    print("=== ATTRIBUTION (Carhart + sector, Newey-West) ===")
    # raw (rebuild on daily grid for apples-to-apples), then best daily variants
    raw_pg = to_period_grid(ds)
    mask = ~np.isnan(raw_pg)
    print("raw daily strategy (aggregated to period grid):")
    L.attribution("raw daily", raw_pg, regs, mask=mask)

    for variant in {best_cap1["variant"], best_overall["variant"]}:
        pg = to_period_grid(daily_series[variant])
        m = ~np.isnan(pg)
        print(f"{variant}:")
        L.attribution(variant, pg, regs, mask=m)

    # also show the library raw period series alpha as the documented baseline
    print("lib raw period series (documented baseline ~+6.3%/yr t=1.71):")
    L.attribution("lib raw period", raw_period, regs)


if __name__ == "__main__":
    main()
