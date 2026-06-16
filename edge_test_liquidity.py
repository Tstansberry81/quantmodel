"""The Edge -- LIQUIDITY / SIZE FLOOR test.

For a SHORT-TERM TRADING product, holdings must be liquid + big enough to trade
real money without moving the market. This script:
  1. Characterizes the unconstrained top-10 book's size profile (are the winners
     tiny illiquid names you couldn't actually trade?).
  2. Applies market-cap floors BEFORE ranking and re-ranks top-10 within the
     floored universe; reports 1Y/2Y/5Y/MAX CAGR/Sharpe/maxDD per floor.
  3. If data/cache/volume.pkl exists, repeats with a dollar-volume floor.
  4. Quantifies the trade-off vs the no-floor baseline.

pit_mcap is in raw USD (median ~$6.8B in mid-sample). Floors expressed in USD.
Run: .venv\\Scripts\\python.exe edge_test_liquidity.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import edge_lib as E
from qmodel import engine

B = 1e9  # one billion USD
M = 1e6  # one million USD


def build_dollar_vol_lookup():
    """Per-rebalance-date dollar-volume (price * avg 21d share volume) keyed by
    company_key. Volume cache columns are already in EXCHANGE_TICKER == company_key
    form. Returns dict[date_ts] -> Series(company_key -> $vol). Volume history
    begins 2018-05; earlier dates get an empty series (no floor applied)."""
    vol = pd.read_pickle(os.path.join("data", "cache", "volume.pkl"))
    vol.index = pd.to_datetime(vol.index)
    avgvol = vol.rolling(21, min_periods=10).mean()          # avg daily shares
    data = engine._load_bt_data()
    # price series per company_key
    prices = {ck: blob["prices"] for ck, blob in data.items()
              if blob.get("prices") is not None}
    return vol, avgvol, prices


def dollar_vol_at(date_ts, avgvol, prices):
    """$-volume Series at a rebalance date for all names with both vol+price."""
    if date_ts < avgvol.index.min():
        return pd.Series(dtype=float)
    vi = avgvol.index.searchsorted(date_ts, side="right") - 1
    if vi < 0:
        return pd.Series(dtype=float)
    shares = avgvol.iloc[vi]                                  # Series by company_key
    out = {}
    for ck, sh in shares.items():
        if np.isnan(sh):
            continue
        pr = prices.get(ck)
        if pr is None:
            continue
        pi = pr.index.searchsorted(date_ts, side="right") - 1
        if pi < 0:
            continue
        px = float(pr.iloc[pi])
        out[ck] = px * sh
    return pd.Series(out)


def dvol_floor_holds(pan, floor_usd, avgvol, prices, n=10, weights=None):
    """Top-n per period from universe filtered to dollar-volume >= floor_usd.
    Periods before volume history exist are left unfiltered (mcap-only book)."""
    holds, applied = [], []
    for i, df in enumerate(pan.panels):
        dv = dollar_vol_at(pan.bdates[i], avgvol, prices)
        if len(dv) == 0:
            holds.append(E.select_topN(df, n, weights)); applied.append(False); continue
        keep = dv[dv >= floor_usd].index
        sub = df[df["company_key"].isin(keep)]
        holds.append(E.select_topN(sub, n, weights)); applied.append(True)
    return holds, np.array(applied)


def floor_holds(pan, floor_usd, n=10, weights=None):
    """Top-n per period from the universe filtered to pit_mcap >= floor_usd."""
    holds = []
    for df in pan.panels:
        sub = df[df["pit_mcap"] >= floor_usd]
        holds.append(E.select_topN(sub, n, weights))
    return holds


def book_size_profile(pan, holds):
    """Per-period median/min pit_mcap of the SELECTED names; aggregate over time."""
    meds, mins, n_below = [], [], {1: 0, 2: 0, 5: 0}
    tot = 0
    for i, df in enumerate(pan.panels):
        sel = df[df["company_key"].isin(holds[i])]
        if not len(sel):
            continue
        mc = sel["pit_mcap"].values
        meds.append(np.median(mc)); mins.append(np.min(mc))
        tot += len(mc)
        for thr in (1, 2, 5):
            n_below[thr] += int((mc < thr * B).sum())
    meds, mins = np.array(meds), np.array(mins)
    return {
        "median_of_period_medians": np.median(meds),
        "median_of_period_mins": np.median(mins),
        "smallest_ever_held": mins.min(),
        "pct_picks_below_1B": 100.0 * n_below[1] / tot,
        "pct_picks_below_2B": 100.0 * n_below[2] / tot,
        "pct_picks_below_5B": 100.0 * n_below[5] / tot,
        "n_picks": tot,
    }


def windows_row_arr(rets, windows=(("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None))):
    out = {}
    rets = np.asarray(rets, float)
    for wname, k in windows:
        kk = len(rets) if k is None else min(k, len(rets))
        c, dd, sh = E.perf(rets[-kk:])
        out[wname] = (c, sh, dd)
    return out


def windows_row(pan, rets, **kw):
    return windows_row_arr(rets, **kw)


def fmt_perf_table(rows, windows=("1Y", "2Y", "5Y", "MAX")):
    """rows = list[(label, windows_dict)]."""
    hdr = f"{'floor':<14}"
    for w in windows:
        hdr += f"{w+' CAGR':>11}{'Sh':>6}{'DD':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label, wd in rows:
        line = f"{label:<14}"
        for w in windows:
            c, sh, dd = wd[w]
            line += f"{c*100:9.1f}% {sh:5.2f}{dd*100:5.0f}%"
        print(line)


def main():
    print("Loading Edge panel (~1-2 min) ...")
    pan = E.load_edge_panel()
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}, "
          f"avg {np.mean([len(p) for p in pan.panels]):.0f} names/period\n")

    # ---- distribution of available universe mcap (calibrate floors) ----------
    all_mc = np.concatenate([df["pit_mcap"].dropna().values for df in pan.panels])
    print("=== Universe pit_mcap distribution (all candidate-periods, USD) ===")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pv = np.nanpercentile(all_mc, qs)
    for q, v in zip(qs, pv):
        print(f"  p{q:<3d} = ${v/B:8.2f}B")
    print(f"  names/period >= $1B : {np.mean([(df['pit_mcap']>=1*B).sum() for df in pan.panels]):.0f}")
    print(f"  names/period >= $2B : {np.mean([(df['pit_mcap']>=2*B).sum() for df in pan.panels]):.0f}")
    print(f"  names/period >= $5B : {np.mean([(df['pit_mcap']>=5*B).sum() for df in pan.panels]):.0f}")
    print(f"  names/period >= $10B: {np.mean([(df['pit_mcap']>=10*B).sum() for df in pan.panels]):.0f}")
    print(f"  names/period >= $25B: {np.mean([(df['pit_mcap']>=25*B).sum() for df in pan.panels]):.0f}\n")

    # ---- TASK 1: baseline book size profile ----------------------------------
    base_holds = [E.select_topN(df) for df in pan.panels]
    base_rets = E.period_returns(pan, base_holds)
    prof = book_size_profile(pan, base_holds)
    print("=== TASK 1: Unconstrained top-10 book size profile ===")
    print(f"  median of per-period MEDIAN selected mcap : ${prof['median_of_period_medians']/B:.2f}B")
    print(f"  median of per-period MIN selected mcap    : ${prof['median_of_period_mins']/B:.2f}B")
    print(f"  smallest name ever held                   : ${prof['smallest_ever_held']/B:.3f}B")
    print(f"  % of all picks below $1B  : {prof['pct_picks_below_1B']:.1f}%")
    print(f"  % of all picks below $2B  : {prof['pct_picks_below_2B']:.1f}%")
    print(f"  % of all picks below $5B  : {prof['pct_picks_below_5B']:.1f}%")
    print(f"  ({prof['n_picks']} total picks)\n")

    # ---- TASK 2: size floors -------------------------------------------------
    floors = [("none", 0.0), ("$1B", 1*B), ("$2B", 2*B), ("$5B", 5*B),
              ("$10B", 10*B), ("$25B", 25*B)]
    print("=== TASK 2: Performance vs market-cap floor (top-10 by ret_63) ===")
    rows = []
    win_by_floor = {}
    for label, fl in floors:
        holds = base_holds if fl == 0.0 else floor_holds(pan, fl)
        rets = base_rets if fl == 0.0 else E.period_returns(pan, holds)
        wd = windows_row(pan, rets)
        win_by_floor[label] = wd
        rows.append((label, wd))
    fmt_perf_table(rows)

    # S&P reference
    spx_wd = windows_row(pan, pan.spxf)
    print()
    fmt_perf_table([("S&P 500", spx_wd)])

    # ---- TASK 4: trade-off vs baseline ---------------------------------------
    print("\n=== TASK 4: Trade-off vs no-floor baseline (CAGR delta, pp) ===")
    base = win_by_floor["none"]
    hdr = f"{'floor':<14}" + "".join(f"{w:>12}" for w in ("1Y", "2Y", "5Y", "MAX"))
    print(hdr)
    for label, _ in floors[1:]:
        wd = win_by_floor[label]
        line = f"{label:<14}"
        for w in ("1Y", "2Y", "5Y", "MAX"):
            d = (wd[w][0] - base[w][0]) * 100
            line += f"{d:>+11.1f}p"
        print(line)

    # ---- TASK 3: dollar-volume floor (the honest liquidity gate) -------------
    vol_path = os.path.join("data", "cache", "volume.pkl")
    print()
    if not os.path.exists(vol_path):
        print("=== TASK 3: data/cache/volume.pkl NOT present ===")
        print("  Dollar-volume floor requires the volume cache. Using mcap floor")
        print("  as proxy; dollar-volume is a FOLLOW-UP once the cache lands.")
        return

    print("=== TASK 3: dollar-volume floor (price * avg 21d volume) ===")
    vol, avgvol, prices = build_dollar_vol_lookup()
    vstart = avgvol.dropna(how="all").index.min()
    # subset of rebalances with volume coverage
    cov = np.array([pan.bdates[i] >= vstart for i in range(pan.T)])
    print(f"  volume coverage: {cov.sum()}/{pan.T} rebalances "
          f"({pan.bdates[cov.argmax()].date()} -> {pan.bdates[-1].date()})")

    # $-volume distribution of the universe over the covered window (calibrate)
    samp = []
    for i in range(pan.T):
        if not cov[i]:
            continue
        dv = dollar_vol_at(pan.bdates[i], avgvol, prices)
        samp.append(dv.values)
    samp = np.concatenate(samp)
    print("  universe daily $-volume distribution (covered periods):")
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<3d} = ${np.nanpercentile(samp, q)/M:8.1f}M/day")

    # baseline restricted to the covered window for an apples-to-apples compare
    base_cov_rets = base_rets[cov]
    dfloors = [("none(cov)", 0.0), ("$5M/day", 5*M), ("$10M/day", 10*M),
               ("$25M/day", 25*M), ("$50M/day", 50*M), ("$100M/day", 100*M)]
    print("\n  Performance vs $-volume floor (window = volume-covered periods only):")
    rows3 = []
    base_dwd = None
    for label, fl in dfloors:
        if fl == 0.0:
            rets = base_cov_rets
        else:
            holds, _ = dvol_floor_holds(pan, fl, avgvol, prices)
            rets = E.period_returns(pan, holds)[cov]
        wd = windows_row_arr(rets)
        if base_dwd is None:
            base_dwd = wd
        rows3.append((label, wd))
    fmt_perf_table(rows3)
    print("\n  Trade-off vs no-floor (covered window), CAGR delta pp:")
    for label, wd in rows3[1:]:
        line = f"  {label:<12}"
        for w in ("1Y", "2Y", "5Y", "MAX"):
            line += f"{(wd[w][0]-base_dwd[w][0])*100:>+11.1f}p"
        print(line)


if __name__ == "__main__":
    main()
