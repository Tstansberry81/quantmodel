"""The Edge -- same-calendar-month seasonality (Heston-Sadka 2008, JFE).

Finding to test: a stock's historical return in a given CALENDAR MONTH predicts its
return in that same month in future years (annual-lag autocorrelation), superimposed
on momentum/reversal. Documented in large caps (top-30% cap: 0.69%/mo, t=4.19),
size-independent, and LOW turnover (cost-cheap) -- a rare combination.

Big advantage over the turnover test: uses our FULL 2005+ history, not the post-2018
volume window. Genuinely orthogonal to the acceleration core.

Signal for name i at rebalance date d (calendar month m = month(d)):
  seas_y1  = monthly return of i in month m, one year ago (the 12-month lag; the
             paper says this single lag captures most of the year-1 momentum spread)
  seas_avg = average of i's month-m returns over annual lags 1..5 (the paper's
             strongest combo is lags 2-5yr, which also SIDESTEPS 12-mo momentum overlap)
  seas_hs  = average over annual lags 2..5 only (cleanest, momentum-orthogonal)

Predicts the next ~1-month return, so tested primarily at hold=21 (the panel's
fwd_ret is ~1 month there). Reuses edge_signal_ic scorecard machinery.

No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)
from edge_signal_ic import spearman, t_stat, resid_of, deciles


def monthly_returns():
    """Per-name month-end total return series (calendar-month), for seasonality lags."""
    data = engine._load_bt_data()
    out = {}
    for ck, b in data.items():
        pr = b.get("prices")
        if pr is None or len(pr) < 300:
            continue
        m = pr.resample("ME").last()
        out[ck] = m.pct_change()          # monthly return indexed by month-end
    return out


def seas_at(mret, d):
    """(seas_y1, seas_avg[1-5], seas_hs[2-5]) for one name at rebalance date d."""
    if mret is None:
        return np.nan, np.nan, np.nan
    cm = d.month
    vals = {}
    for y in range(1, 6):
        yr = d.year - y
        # month-end of (yr, cm): find the mret entry in that year+month
        hits = mret[(mret.index.year == yr) & (mret.index.month == cm)]
        if len(hits) and np.isfinite(hits.iloc[-1]):
            vals[y] = float(hits.iloc[-1])
    y1 = vals.get(1, np.nan)
    avg15 = np.mean(list(vals.values())) if vals else np.nan
    hs = [vals[y] for y in (2, 3, 4, 5) if y in vals]
    hs25 = np.mean(hs) if hs else np.nan
    return y1, avg15, hs25


def build(pan, mret):
    panels = []
    for i, df in enumerate(pan.panels):
        d = pan.bdates[i]
        recs = {}
        for ck in df["company_key"]:
            recs[ck] = seas_at(mret.get(ck), d)
        add = pd.DataFrame.from_dict(recs, orient="index",
                                     columns=["seas_y1", "seas_avg", "seas_hs"])
        panels.append(df.set_index("company_key").join(add).reset_index())
        if (i + 1) % 60 == 0:
            print(f"    seasonality {i+1}/{pan.T}")
    return panels


def report(panels, ppy, col):
    ic = np.array([spearman(df[col], df["fwd_ret"]) for df in panels], float)
    inc = np.array([spearman(resid_of(df[col].to_numpy(), df["accel"].to_numpy()),
                             df["fwd_ret"].to_numpy()) for df in panels], float)
    dm, mono, spread = deciles(panels, col)
    print(f"   {col:<9} IC {np.nanmean(ic)*100:+6.2f}%  t={t_stat(ic):+5.2f}   "
          f"mono {mono:+.2f}   D10-D1 {spread*100:+5.2f}%   "
          f"incrIC(vs accel) {np.nanmean(inc)*100:+5.2f}% t={t_stat(inc):+5.2f}")


def main():
    print("Building monthly return series ...")
    mret = monthly_returns()
    for hold in (21, 42):
        pan = E.load_edge_panel(hold=hold)
        print(f"\n{'='*80}\nhold={hold} — {pan.T} rebalances "
              f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()} (FULL history)\n{'='*80}")
        panels = build(pan, mret)
        print("\nSeasonality signals [Heston-Sadka: positive IC; seas_hs is momentum-orthogonal]")
        for col in ("seas_y1", "seas_avg", "seas_hs"):
            report(panels, pan.ppy, col)
        print("   (reference) accel/reversal baselines:")
        for col in ("accel", "ret_21"):
            report(panels, pan.ppy, col)
    print("\nDONE.  (hold to t>3 on our universe per Harvey-Liu-Zhu; seas_hs is the clean one)")


if __name__ == "__main__":
    main()
