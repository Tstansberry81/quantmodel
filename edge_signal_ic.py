"""The Edge -- bias-resistant signal diagnostics (price-only).

Hunt for NEW alpha the honest way: score each candidate signal on the whole
cross-section, every rebalance, BEFORE it ever touches a top-10 backtest. A
top-10 CAGR over a few overlapping windows is trivial to overfit; a rank-IC over
~800 names x ~230 dates is not.

For every signal we report:
  * mean rank-IC + t-stat + information ratio      (does it rank forward returns?)
  * decile staircase + monotonicity + top-bottom    (is the effect ordered, or one lucky bin?)
  * incremental IC after removing `accel`           (does it ADD to what we already trade?)
  * first-half / second-half / ex-2020 IC           (is it stable, or regime-fitted?)

All candidates are computed from daily TOTAL-RETURN PRICES ONLY (2005+), matching
the market-data-only philosophy. Nothing here modifies edge_lib.py or qmodel/;
a signal graduates into the panel + score() weights only after it clears the bar.

No statsmodels in the venv -> stats done by hand in numpy.

Usage:
    .venv/Scripts/python.exe edge_signal_ic.py            # hold=21 (product clock)
    .venv/Scripts/python.exe edge_signal_ic.py --hold 42  # re-check at the 2M clock
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)

LB = E.LB                      # 251-day trailing window (matches the panel)
FORM_HI, FORM_LO = 126, 21     # momentum formation = t-126 .. t-21 (skip last month)


# ---- rank / correlation helpers (Spearman = Pearson of ranks) ---------------
def _rank(a):
    return pd.Series(a).rank().to_numpy()


def spearman(x, y):
    """Spearman rank correlation, NaN-safe. Returns np.nan if <5 valid pairs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    rx, ry = _rank(x[ok]), _rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def resid_of(y, x):
    """Cross-sectional OLS residual of y on x (with intercept), NaN-safe.
    Used to ask 'what does this signal add AFTER accel?'."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    out = np.full_like(y, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or x[ok].std() == 0:
        return out
    b = np.polyfit(x[ok], y[ok], 1)
    out[ok] = y[ok] - (b[0] * x[ok] + b[1])
    return out


# ---- build the price-only candidate signals, aligned to the panel -----------
def _raw_prep():
    """Rebuild the per-name daily arrays (mirrors load_edge_panel's prep) so we
    can compute window-based signals the panel discards. Price-only."""
    data = engine._load_bt_data(); bm = engine._benchmarks_cached()
    spx = bm["SP500"].dropna(); mret_full = spx.pct_change()
    prep = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB + 60:
            continue
        arr = np.asarray(pr.values, float)
        pidx = np.asarray(pr.index.values, "datetime64[ns]")
        rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:] / arr[:-1] - 1
        mret = np.asarray(mret_full.reindex(pr.index, method="ffill").values, float)
        prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret}
    return prep


def _signals_at(P, pos):
    """All price-only candidate signals for one name at trailing position `pos`.
    Everything uses data through `pos` only -- no lookahead."""
    arr = P["arr"]; w = P["rets"][pos - LB + 1: pos + 1]; wm = P["mret"][pos - LB + 1: pos + 1]
    ok = np.isfinite(w) & np.isfinite(wm)
    out = {}

    # market-model residuals (idiosyncratic path): r - (alpha + beta*mkt)
    resid = np.full_like(w, np.nan)
    if ok.sum() > 60 and wm[ok].var() > 0:
        beta = np.cov(w[ok], wm[ok])[0, 1] / wm[ok].var()
        alpha = w[ok].mean() - beta * wm[ok].mean()
        resid[ok] = w[ok] - (alpha + beta * wm[ok])
    rstd = np.nanstd(resid)

    # 1. RESIDUAL MOMENTUM (Blitz-Huij-Martens): standardized cumulative idio
    #    return over the 12-1-style formation window, beta- AND alpha-neutral.
    form = resid[-FORM_HI:-FORM_LO]
    out["resid_mom"] = float(np.nansum(form) / rstd) if rstd > 0 else np.nan

    # 2. INFORMATION DISCRETENESS (Da-Gurun-Warachka frog-in-the-pan) over 63d.
    #    ID = sign(pret)*(%down - %up). Continuous (low ID) winners continue.
    w63 = w[-63:]; w63 = w63[np.isfinite(w63)]
    if len(w63) > 10:
        pret = np.prod(1 + w63) - 1
        fneg = np.mean(w63 < 0); fpos = np.mean(w63 > 0)
        idisc = np.sign(pret) * (fneg - fpos)
        out["id_disc"] = float(idisc)                      # expect NEGATIVE IC
        out["sm_mom"] = float(pret * -idisc)               # smooth-momentum interaction
    else:
        out["id_disc"] = np.nan; out["sm_mom"] = np.nan

    # 3. MAX effect (Bali-Cakici-Whitelaw): biggest 1-day return last month.
    w21 = w[-21:]
    out["maxret"] = float(np.nanmax(w21)) if np.isfinite(w21).any() else np.nan  # expect NEG IC

    # 4. IDIOSYNCRATIC VOL (Ang et al.): residual vol last month, annualized.
    out["ivol"] = float(np.nanstd(resid[-21:]) * np.sqrt(252)) if rstd > 0 else np.nan  # expect NEG

    # 5. VOL-SCALED MOMENTUM (ramom): 3-mo return / realized vol.
    v21 = float(np.nanstd(w21, ddof=1) * np.sqrt(252)) if np.isfinite(w21).sum() > 2 else np.nan
    ret63 = arr[pos] / arr[pos - 63] - 1
    out["ramom"] = float(ret63 / v21) if v21 and v21 > 0 else np.nan
    return out


def build_signals(pan):
    """Attach the price-only candidate columns to a COPY of each panel df,
    aligned by (rebalance date, company_key). Reuses the panel's own fwd_ret."""
    prep = _raw_prep()
    panels = []
    for i, df in enumerate(pan.panels):
        dt = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        recs = {}
        for ck in df["company_key"]:
            P = prep.get(ck)
            if P is None:
                continue
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB or pos >= len(P["arr"]):
                continue
            recs[ck] = _signals_at(P, pos)
        add = pd.DataFrame.from_dict(recs, orient="index")
        out = df.set_index("company_key").join(add).reset_index()
        panels.append(out)
        if (i + 1) % 40 == 0:
            print(f"    built signals {i+1}/{len(pan.panels)}")
    return panels


# ---- the honest scorecard ----------------------------------------------------
def ic_series(panels, col):
    """Per-rebalance rank-IC of `col` vs forward return."""
    return np.array([spearman(df[col], df["fwd_ret"]) for df in panels], float)


def incremental_ic(panels, col, base="accel"):
    """Rank-IC of the part of `col` that is orthogonal to `base` (cross-sectional
    residual). Tells us if the signal ADDS to acceleration or just echoes it."""
    ics = []
    for df in panels:
        r = resid_of(df[col].to_numpy(), df[base].to_numpy())
        ics.append(spearman(r, df["fwd_ret"].to_numpy()))
    return np.array(ics, float)


def overlap(panels, col, base="accel"):
    """Average |cross-sectional Spearman(col, accel)| -- how much it echoes accel."""
    cs = [spearman(df[col], df[base]) for df in panels]
    return float(np.nanmean(np.abs(cs)))


def deciles(panels, col, nd=10):
    """Average forward return by cross-sectional decile of `col` (per-period
    deciles, then averaged). Returns (decile_means, monotonicity, top_minus_bottom)."""
    rows = []
    for df in panels:
        s = pd.to_numeric(df[col], errors="coerce")
        f = pd.to_numeric(df["fwd_ret"], errors="coerce")
        m = s.notna() & f.notna()
        if m.sum() < nd * 3:
            continue
        try:
            q = pd.qcut(s[m].rank(method="first"), nd, labels=False)
        except ValueError:
            continue
        rows.append(f[m].groupby(q).mean())
    if not rows:
        return np.full(nd, np.nan), np.nan, np.nan
    dm = pd.concat(rows, axis=1).mean(axis=1).reindex(range(nd)).to_numpy()
    mono = spearman(np.arange(nd), dm)
    return dm, mono, float(dm[-1] - dm[0])


def t_stat(ic):
    ic = ic[np.isfinite(ic)]
    if len(ic) < 3 or ic.std() == 0:
        return np.nan
    return float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))


def scorecard(panels, bdates, cols):
    yrs = np.array([d.year for d in bdates])
    half = len(panels) // 2
    print("\n" + "=" * 118)
    print(f"{'signal':<12}{'IC':>8}{'t(IC)':>8}{'IR':>7}{'mono':>7}{'D10-D1':>9}"
          f"{'incrIC':>9}{'t(incr)':>9}{'|corr|acc':>11}{'IC 1st':>8}{'IC 2nd':>8}{'IC ex20':>9}")
    print("-" * 118)
    results = {}
    for c in cols:
        if c not in panels[0].columns and c != "ret_21_rev":
            continue
        ic = ic_series(panels, c)
        dm, mono, spread = deciles(panels, c)
        inc = incremental_ic(panels, c) if c != "accel" else ic
        ic1 = np.nanmean(ic[:half]); ic2 = np.nanmean(ic[half:])
        ex20 = np.nanmean(ic[yrs != 2020])
        results[c] = {"ic": np.nanmean(ic), "t": t_stat(ic), "mono": mono,
                      "incr": np.nanmean(inc), "t_incr": t_stat(inc), "deciles": dm}
        print(f"{c:<12}{np.nanmean(ic)*100:>7.2f}%{t_stat(ic):>8.2f}"
              f"{np.nanmean(ic)/np.nanstd(ic) if np.nanstd(ic)>0 else 0:>7.2f}"
              f"{mono:>7.2f}{spread*100:>8.2f}%"
              f"{np.nanmean(inc)*100:>8.2f}%{t_stat(inc):>9.2f}"
              f"{overlap(panels, c):>11.2f}"
              f"{ic1*100:>7.2f}%{ic2*100:>7.2f}%{ex20*100:>8.2f}%")
    print("-" * 118)
    print("Reading it: IC = mean rank-corr(signal, fwd ret). |t|>~2 is real. mono=+1 clean staircase.")
    print("incrIC = IC of the signal AFTER removing accel (what it ADDS). |corr|acc = overlap w/ accel.")
    print("IC 1st/2nd/ex20 = stability. A signal that's strong overall but 0 in one half is regime-fit.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=21)
    args = ap.parse_args()

    print(f"Loading hold={args.hold} panel ...")
    pan = E.load_edge_panel(hold=args.hold)
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}, "
          f"avg {np.mean([len(p) for p in pan.panels]):.0f} names")
    print("Building price-only candidate signals (one-time) ...")
    panels = build_signals(pan)

    # baselines (already in the panel) + new price-only candidates
    baselines = ["accel", "ret_63", "ret_21", "ret_126", "hi_252", "rs_63", "vol_21"]
    candidates = ["resid_mom", "id_disc", "sm_mom", "maxret", "ivol", "ramom"]
    print("\nBASELINES (already computed by the model):")
    scorecard(panels, pan.bdates, baselines)
    print("\nNEW PRICE-ONLY CANDIDATES:")
    scorecard(panels, pan.bdates, candidates)


if __name__ == "__main__":
    main()
