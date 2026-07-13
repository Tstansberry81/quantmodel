"""The Edge -- turnover-conditioned short-term momentum (Medhat-Schmeling 2022, RFS).

The finding to test: double-sort last-month return x share turnover. HIGH-turnover
recent winners CONTINUE (short-term momentum); LOW-turnover recent winners REVERSE.
This matters because our own R1 IC hunt found the ONLY significant 1-month effect is
reversal (ret_21 IC -2.2%, t-2.4) -- Medhat-Schmeling says that sign FLIPS on turnover,
which we never conditioned on. Effect is documented strongest in large, liquid,
high-coverage names (our universe) and to survive costs.

Turnover_i = (price_i * avg 21d share volume_i) / market_cap_i  -- a share-turnover
proxy computable from volume.pkl (2018+) + the panel's pit_mcap. So this is a
POST-2018 test (~7yr); treat as a shorter-sample confirmation, not full-history.

Tests (all cross-sectional, t-stats over rebalances, reusing edge_signal_ic helpers):
  1. IC of ret_21 within LOW / MID / HIGH turnover terciles  (expect neg -> pos gradient)
  2. IC of accel  within the same terciles                    (does our signal sharpen?)
  3. A tradable STM signal = ret_21 gated to the top turnover tercile: its IC, decile
     staircase, and INCREMENTAL IC after removing accel (does it ADD to the product?)
  4. Same at hold=21 (native horizon) and hold=42 (product clock).

No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)
from edge_signal_ic import spearman, t_stat, resid_of


def load_turnover():
    """avg-21d share volume (rolling) + per-name price series, for turnover calc."""
    vol = pd.read_pickle(os.path.join("data", "cache", "volume.pkl"))
    vol.index = pd.to_datetime(vol.index)
    avgvol = vol.rolling(21, min_periods=10).mean()
    data = engine._load_bt_data()
    prices = {ck: b["prices"] for ck, b in data.items() if b.get("prices") is not None}
    return avgvol, prices


def turnover_at(date_ts, avgvol, prices, mcap_series):
    """Turnover Series at a rebalance date: $vol / mcap, keyed by company_key."""
    if date_ts < avgvol.index.min():
        return pd.Series(dtype=float)
    vi = avgvol.index.searchsorted(date_ts, side="right") - 1
    if vi < 0:
        return pd.Series(dtype=float)
    shares = avgvol.iloc[vi]
    out = {}
    for ck, sh in shares.items():
        if np.isnan(sh) or ck not in mcap_series.index:
            continue
        mc = mcap_series.get(ck)
        pr = prices.get(ck)
        if pr is None or mc is None or not np.isfinite(mc) or mc <= 0:
            continue
        pi = pr.index.searchsorted(date_ts, side="right") - 1
        if pi < 0:
            continue
        out[ck] = (float(pr.iloc[pi]) * sh) / mc          # dollar volume / mcap
    return pd.Series(out)


def build(pan, avgvol, prices):
    """Attach a `turnover` column to each panel df (post-2018 rebalances only)."""
    panels, dates = [], []
    for i, df in enumerate(pan.panels):
        d = pan.bdates[i]
        if d < avgvol.index.min():
            continue
        mc = df.set_index("company_key")["pit_mcap"]
        tov = turnover_at(d, avgvol, prices, mc)
        if len(tov) < 50:
            continue
        out = df.set_index("company_key")
        out["turnover"] = tov
        panels.append(out.reset_index().dropna(subset=["turnover"]))
        dates.append(d)
    return panels, pd.DatetimeIndex(dates)


def tercile_ic(panels, sig):
    """Mean IC of `sig` vs fwd_ret within low/mid/high turnover terciles."""
    lo, mid, hi = [], [], []
    for df in panels:
        t = pd.to_numeric(df["turnover"], errors="coerce")
        q1, q2 = t.quantile(1/3), t.quantile(2/3)
        for mask, bucket in ((t <= q1, lo), ((t > q1) & (t <= q2), mid), (t > q2, hi)):
            sub = df[mask]
            if len(sub) >= 15:
                bucket.append(spearman(sub[sig], sub["fwd_ret"]))
    f = lambda b: (np.nanmean(b), t_stat(np.array(b)), len(b))
    return f(lo), f(mid), f(hi)


def main():
    avgvol, prices = load_turnover()
    for hold in (21, 42):
        pan = E.load_edge_panel(hold=hold)
        panels, dates = build(pan, avgvol, prices)
        if not panels:
            print(f"hold={hold}: no post-2018 rebalances with volume — skip"); continue
        print(f"\n{'='*78}\nhold={hold} — {len(panels)} rebalances "
              f"{dates[0].date()} -> {dates[-1].date()} (post-2018, ~{len(panels)*hold/252:.0f}yr)\n{'='*78}")

        print("\n1. IC of ret_21 (last-month return) by turnover tercile "
              "[Medhat-Schmeling: neg in LOW, pos in HIGH]")
        for name, (ic, t, k) in zip(("LOW ", "MID ", "HIGH"), tercile_ic(panels, "ret_21")):
            print(f"   {name} turnover: IC {ic*100:+6.2f}%  t={t:+5.2f}  (n={k})")

        print("\n2. IC of accel by turnover tercile [does our signal sharpen in high-turnover?]")
        for name, (ic, t, k) in zip(("LOW ", "MID ", "HIGH"), tercile_ic(panels, "accel")):
            print(f"   {name} turnover: IC {ic*100:+6.2f}%  t={t:+5.2f}  (n={k})")

        # 3. tradable STM signal: ret_21 within the top turnover tercile
        print("\n3. Tradable STM = ret_21 among HIGH-turnover names")
        ics, incs, dspreads = [], [], []
        for df in panels:
            t = pd.to_numeric(df["turnover"], errors="coerce")
            hi = df[t > t.quantile(2/3)].copy()
            if len(hi) < 20:
                continue
            ics.append(spearman(hi["ret_21"], hi["fwd_ret"]))
            incs.append(spearman(resid_of(hi["ret_21"].to_numpy(), hi["accel"].to_numpy()),
                                 hi["fwd_ret"].to_numpy()))
            # top-quintile-by-ret21 minus bottom-quintile fwd return, within high-turnover
            r = pd.to_numeric(hi["ret_21"], errors="coerce")
            top = hi[r >= r.quantile(0.8)]["fwd_ret"].mean()
            bot = hi[r <= r.quantile(0.2)]["fwd_ret"].mean()
            dspreads.append(float(top - bot))
        ics, incs, dspreads = map(lambda a: np.array(a, float), (ics, incs, dspreads))
        ppy = pan.ppy
        print(f"   IC(ret_21 | high-turnover):        {np.nanmean(ics)*100:+6.2f}%  t={t_stat(ics):+5.2f}")
        print(f"   incremental IC after removing accel:{np.nanmean(incs)*100:+6.2f}%  t={t_stat(incs):+5.2f}")
        print(f"   top-minus-bottom quintile fwd ret:  {np.nanmean(dspreads)*100:+6.2f}%/period "
              f"(~{np.nanmean(dspreads)*ppy*100:+.1f}%/yr)  t={t_stat(dspreads):+5.2f}")

    print("\nDONE.  (post-2018 sample; hold to t>3 on our universe per Harvey-Liu-Zhu)")


if __name__ == "__main__":
    main()
