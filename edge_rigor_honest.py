"""Reconciled rigor: DSR and PBO on ONE trial set, ONE (honest) basis (queue #3).

The committed edge_rigor.py had inconsistent inputs: the Deflated Sharpe read a
phantom ~150 trial count + a stale CSV for Sharpe dispersion and ran on the OLD
single-sleeve/period path, while PBO used a separate 24-config grid. This runs
BOTH against the SAME reference trial set (the honest config resweep,
edge_configsweep_honest.csv) on the SAME production basis (staggered / t+1 /
daily / continuous 200dMA regime, net):

  * Newey-West t  -- on the product's DAILY excess vs S&P.
  * Deflated Sharpe -- product daily series; n_trials = size of the honest sweep
    grid actually searched; Sharpe dispersion (var_sr) from that same grid.
  * PBO (CSCV)   -- over the honest n x mix grid at the product hold, each config's
    DAILY return series (same basis + same trial family as the DSR inputs).

Run AFTER edge_test_configsweep_honest.py. Usage: .venv/Scripts/python.exe edge_rigor_honest.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as D
from edge_rigor import newey_west_t, deflated_sharpe, pbo_cscv

S = E.EDGE_SPEC
SWEEP = "data/cache/edge_configsweep_honest.csv"


def honest_daily(hold, n, mix):
    idx, model, spx, ndx, to, prim = E._edge_daily(
        hold, n, S["mcap_floor"], S["corr_cap"], S["corr_lookback"], S["regime_expo"],
        S["cost_bps"], (("accel", 1.0),), mix, S["growth_thresh"], True, True)
    return pd.Series(model, index=idx), pd.Series(spx, index=idx)


def to_periods(daily):
    """Non-overlapping hold-day period returns from a daily series. NW-t and CSCV
    assume weakly-dependent observations; the daily series holds each book 42 days
    (heavy overlap autocorrelation), so significance tests MUST use these ~124
    non-overlapping period returns, not the 5,250 overlapping daily ones."""
    r = daily.dropna().to_numpy(float)
    hold = S["hold"]
    eq = np.cumprod(1 + r)
    idx = list(range(0, len(eq), hold))
    lvl = eq[idx]
    return lvl[1:] / lvl[:-1] - 1


def main():
    hold = S["hold"]; ppy = 252.0 / hold
    model, spx = honest_daily(hold, 10, 0.75)

    sw = pd.read_csv(SWEEP)
    ann_sr = sw[sw.window == "MAX"].sharpe.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n_trials = int(len(ann_sr))
    var_sr = float(np.var(ann_sr / np.sqrt(ppy)))     # annual -> per-PERIOD Sharpe variance

    print("=" * 78)
    print(f"RECONCILED RIGOR — product (hold={hold}, n=10, mix=0.75), honest basis")
    print(f"Single trial set: honest resweep -> {n_trials} configs; ann-Sharpe "
          f"{ann_sr.min():.2f}..{ann_sr.max():.2f}. Stats on NON-OVERLAPPING {hold}-day periods.")
    print("=" * 78)

    # ---- period (non-overlapping) returns of the honest product ----
    mp, sp = to_periods(model), to_periods(spx)
    k = min(len(mp), len(sp)); ex = mp[:k] - sp[:k]
    t_nw, T, L = newey_west_t(ex)
    t_raw = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
    print(f"\nNewey-West t on excess vs S&P (per {hold}d period, T={T}, L={L}): "
          f"{t_nw:+.2f}  raw {t_raw:+.2f}  ({'significant' if abs(t_nw) > 2 else 'NOT sig'})")

    dsr, sr, sr_star, g3, g4, _ = deflated_sharpe(mp, n_trials, var_sr)
    print(f"\nDeflated Sharpe ({n_trials}-trial set, per-period):")
    print(f"  per-period SR {sr:.3f} (~ann {sr*np.sqrt(ppy):.2f}); skew {g3:+.2f}, kurt {g4:.1f}; "
          f"SR* {sr_star:.3f}")
    print(f"  DSR = {dsr:.3f}   ({'PASSES' if dsr > 0.95 else 'FAILS'} >0.95)")

    grid = [(n, m) for n in (5, 6, 7, 8, 9, 10) for m in (0.0, 0.25, 0.5, 0.75, 1.0)]
    cols = []
    for n, m in grid:
        md, _ = honest_daily(hold, n, m)
        cols.append(pd.Series(to_periods(md)))
    R = pd.concat(cols, axis=1).dropna().to_numpy(float)
    pbo, _ = pbo_cscv(R, S_blocks=8)
    print(f"\nPBO (CSCV over {R.shape[1]} n×mix configs, {R.shape[0]} non-overlapping periods):")
    print(f"  PBO = {pbo:.2f}   ({'OVERFIT risk high' if pbo > 0.5 else 'acceptable'} — <0.5 good)")

    # for contrast, the (invalid) daily-basis numbers that overlap-autocorrelation inflates
    td, Td, Ld = newey_west_t((model - spx).to_numpy(float))
    Rd = pd.concat([honest_daily(hold, n, m)[0] for n, m in grid], axis=1).dropna().to_numpy(float)
    pbod, _ = pbo_cscv(Rd, S_blocks=8)
    print(f"\n[contrast — daily basis, statistically INVALID for these tests due to 42-day "
          f"overlap autocorrelation: NW-t {td:+.2f} (L={Ld}), PBO {pbod:.2f}]")
    print("\nBoth DSR and PBO now share one honest trial set + basis; significance on "
          "non-overlapping periods.\nDONE.")


if __name__ == "__main__":
    main()
