"""Neutralization / alpha-attribution test for THE EDGE (short-term acceleration model).

Mirror of the long-term model's tech_bias_test, adapted to the Edge:
does the Edge produce REAL selection alpha, or is it just harvesting the
momentum / acceleration risk premium you could buy passively?

Because the Edge IS an acceleration-momentum strategy, the decisive control is
an in-universe ACCELERATION long-short factor: if alpha dies once you control
for the acceleration premium itself, the Edge is a leveraged factor bet, not skill.

Steps:
  1. Build in-universe factor returns at the Edge's OWN 42-day frequency from
     pan.panels (long-short terciles of fwd_ret):
        MKT   = spxf - RF_PER
        SMB   = small-cap minus big-cap tercile (by pit_mcap)
        MOM   = high ret_63 minus low (generic momentum premium)
        ACCEL = high accel minus low (the strategy's OWN signal premium)
  2. Regress the Edge's NET excess return on increasing controls (Newey-West HAC):
        vs MKT
        vs MKT + SMB + MOM
        vs MKT + SMB + MOM + ACCEL   <- decisive test
  3. External cross-check: Edge daily series -> calendar months -> real Ken-French
     Carhart factors (Mkt-RF/SMB/HML/UMD).
  4. Compare to the long-term model's findings.

Does NOT modify edge_lib.py / tech_bias_lib.py / qmodel.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import edge_lib as E
import tech_bias_lib as TB
import tech_bias_external as EXT

# The Edge full spec
HOLD = 42
N = 20
MCAP_FLOOR = 2e9
CORR_CAP = 0.50
CORR_LB = 126
REGIME_EXPO = 0.25
COST_BPS = 10.0
SIGNAL = {"accel": 1.0}

# IMPORTANT: hold=42 => 252/42 = 6 rebalances/yr. E.PPY (=12, for the 21-day
# default) is WRONG for this frequency and would double-count the annualization.
PPY = 252.0 / HOLD            # = 6.0   (empirically confirmed: 6.04 periods/yr)
RF_PER = E.RF_PER             # risk-free per-period (already at the right per-period scale)


def ann(a_per: float) -> float:
    return (1.0 + a_per) ** PPY - 1.0


def ls_factor(df: pd.DataFrame, col: str, higher_is_long: bool = True) -> float:
    """Equal-weight long-short top-tercile minus bottom-tercile of fwd_ret,
    ranked by `col`. (Same construction as TB._ls_factor.)"""
    g = df.dropna(subset=[col, "fwd_ret"])
    if len(g) < 15:
        return np.nan
    v = pd.to_numeric(g[col], errors="coerce")
    f = g["fwd_ret"].to_numpy(float)
    q1, q2 = v.quantile(1 / 3), v.quantile(2 / 3)
    hi = f[(v >= q2).to_numpy()]
    lo = f[(v <= q1).to_numpy()]
    if len(hi) == 0 or len(lo) == 0:
        return np.nan
    long_leg, short_leg = (hi, lo) if higher_is_long else (lo, hi)
    return float(np.nanmean(long_leg) - np.nanmean(short_leg))


def build_factors(pan):
    smb, mom, accel = [], [], []
    for df in pan.panels:
        smb.append(ls_factor(df, "pit_mcap", higher_is_long=False))   # small minus big
        mom.append(ls_factor(df, "ret_63", higher_is_long=True))      # generic 3m momentum
        accel.append(ls_factor(df, "accel", higher_is_long=True))     # the Edge's OWN signal
    return {
        "MKT": pan.spxf - RF_PER,
        "SMB": np.nan_to_num(np.array(smb)),
        "MOM": np.nan_to_num(np.array(mom)),
        "ACCEL": np.nan_to_num(np.array(accel)),
    }


def regress(y, regressors: dict, L: int = 6):
    names = ["alpha"] + list(regressors)
    X = np.column_stack([np.ones(len(y))] + [regressors[k] for k in regressors])
    beta, se, t = TB.ols_nw(y, X, L=L)
    resid = y - X @ beta
    ss_res = float(resid @ resid); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, names, r2


def step(label, y, regressors):
    beta, se, t, names, r2 = regress(y, regressors)
    a = ann(beta[0])
    tag = "** survives (t>2)" if abs(t[0]) > 2 else ("NOT signif (t<1.65)" if abs(t[0]) < 1.65 else "~ marginal")
    print(f"\n  [{label}]")
    print(f"     alpha {a*100:+.2f}%/yr   (per-period {beta[0]*100:+.3f}%)   t={t[0]:+.2f}   {tag}")
    for i, nm in enumerate(names[1:], 1):
        print(f"        {nm:<8} loading={beta[i]:+.3f}   t={t[i]:+.2f}")
    print(f"     R^2 = {r2:.3f}")
    return a, float(t[0]), {nm: (float(beta[i]), float(t[i])) for i, nm in enumerate(names)}


def main():
    print("=" * 80)
    print("THE EDGE -- alpha neutralization test (acceleration-momentum trading model)")
    print(f"spec: accel signal, hold={HOLD}, n={N}, >=${MCAP_FLOOR/1e9:.0f}B, "
          f"corr-cap {CORR_CAP}, 200dMA regime {REGIME_EXPO}, {COST_BPS:.0f}bps  | PPY={PPY:.0f}")
    print("=" * 80)

    pan = E.load_edge_panel(hold=HOLD)

    # --- Edge full-spec NET per-period returns, aligned to bdates ---
    bdates, gross, net, turn, spxf, ndxf, _holds = E._edge_full(
        HOLD, N, MCAP_FLOOR, CORR_CAP, CORR_LB, REGIME_EXPO, COST_BPS,
        tuple(sorted(SIGNAL.items())))
    print(f"\n  panel: {pan.T} rebalances  {bdates[0].date()} -> {bdates[-1].date()}  "
          f"(~{PPY:.0f}/yr)")
    cg = np.cumprod(1 + np.nan_to_num(net))[-1] ** (PPY / len(net)) - 1
    cs = np.cumprod(1 + np.nan_to_num(spxf))[-1] ** (PPY / len(spxf)) - 1
    print(f"  Edge NET CAGR {cg*100:.1f}%   S&P {cs*100:.1f}%   "
          f"mean net excess/period {(net - RF_PER).mean()*100:+.2f}%")

    # --- in-universe factors at 42-day frequency ---
    F = build_factors(pan)
    print("\n  in-universe factor mean returns/period (annualized):")
    for k, v in F.items():
        print(f"     {k:<6} {v.mean()*100:+.3f}%/period   ({ann(v.mean())*100:+.1f}%/yr)")

    y = net - RF_PER   # Edge NET excess return

    print("\n" + "-" * 80)
    print("STEP-DOWN ATTRIBUTION (Edge NET excess ~ controls, Newey-West HAC)")
    print("-" * 80)
    r_mkt = step("vs MKT only", y, {"MKT": F["MKT"]})
    r_3f = step("vs MKT + SMB + MOM", y, {"MKT": F["MKT"], "SMB": F["SMB"], "MOM": F["MOM"]})
    r_4f = step("vs MKT + SMB + MOM + ACCEL  <-- DECISIVE", y,
                {"MKT": F["MKT"], "SMB": F["SMB"], "MOM": F["MOM"], "ACCEL": F["ACCEL"]})

    # --- external cross-check: real Ken-French Carhart ---
    print("\n" + "-" * 80)
    print("EXTERNAL CROSS-CHECK -- real Ken-French Carhart (calendar-monthly)")
    print("  caveat: daily_from_holdings uses full-spec HOLDINGS but OMITS the")
    print("  regime-cash overlay and the transaction-cost drag (gross of both).")
    print("-" * 80)
    carhart_alpha = carhart_t = umd_load = umd_t = None
    try:
        # rebuild the full-spec holdings exactly as _edge_full does
        holds, prev = [], None
        for i, df in enumerate(pan.panels):
            d = df[df["pit_mcap"] >= MCAP_FLOOR] if MCAP_FLOOR else df
            cks = E.corr_cap_select(d, asof=pan.bdates[i], n=N, weights=SIGNAL,
                                    cap=CORR_CAP, lookback=CORR_LB)
            holds.append(cks)
        daily = E.daily_from_holdings(pan, holds, hold=HOLD)
        strat_m = (1.0 + daily).resample("ME").prod() - 1.0
        strat_m.index = strat_m.index.to_period("M")
        strat_m = strat_m[strat_m != 0.0].dropna()

        try:
            fac, src = EXT.load_french_monthly()
        except Exception as e:
            print(f"  [French download failed: {repr(e)[:70]}] -> ETF proxy")
            fac, src = EXT.load_etf_proxy_monthly()
        print(f"  factor source: {src}")

        common = strat_m.index.intersection(fac.index).sort_values()
        s = strat_m.reindex(common); Fm = fac.reindex(common)
        ym = (s - Fm["RF"]).to_numpy(float)
        reg4 = {"Mkt-RF": Fm["Mkt-RF"].to_numpy(float), "SMB": Fm["SMB"].to_numpy(float),
                "HML": Fm["HML"].to_numpy(float), "UMD": Fm["UMD"].to_numpy(float)}
        beta, se, t, names, r2 = regress(ym, reg4)
        carhart_alpha = (1 + beta[0]) ** 12 - 1; carhart_t = float(t[0])
        umd_load = float(beta[4]); umd_t = float(t[4])
        tag = "** survives (t>2)" if abs(t[0]) > 2 else ("NOT signif (t<1.65)" if abs(t[0]) < 1.65 else "~ marginal")
        print(f"  aligned months: {common.min()} .. {common.max()}  (n={len(common)})")
        print(f"  alpha = {beta[0]*100:+.3f}%/mo   ANNUALIZED {carhart_alpha*100:+.2f}%/yr   t={t[0]:+.2f}   {tag}")
        for i, nm in enumerate(names[1:], 1):
            print(f"      {nm:<8} loading={beta[i]:+.3f}   t={t[i]:+.2f}")
        print(f"  R^2 = {r2:.3f}")
    except Exception as e:
        print(f"  SKIPPED external cross-check: {repr(e)[:100]}")

    # --- verdict vs long-term model ---
    print("\n" + "=" * 80)
    print("VERDICT -- the Edge vs the long-term model")
    print("=" * 80)
    print(f"  LONG-TERM model:  in-universe full-control alpha +6.30%/yr t=1.71 (marginal)")
    print(f"                    external Carhart                +9.50%/yr t=2.89 (survives)")
    print(f"  EDGE  in-univ MKT only          {r_mkt[0]*100:+6.2f}%/yr  t={r_mkt[1]:+.2f}")
    print(f"  EDGE  in-univ MKT+SMB+MOM       {r_3f[0]*100:+6.2f}%/yr  t={r_3f[1]:+.2f}")
    print(f"  EDGE  in-univ +ACCEL (decisive) {r_4f[0]*100:+6.2f}%/yr  t={r_4f[1]:+.2f}")
    if carhart_alpha is not None:
        print(f"  EDGE  external Carhart          {carhart_alpha*100:+6.2f}%/yr  t={carhart_t:+.2f}   "
              f"(UMD loading {umd_load:+.2f}, t={umd_t:+.2f})")
    print()
    accel_load = r_4f[2]["ACCEL"]
    print(f"  ACCEL loading in the decisive regression: {accel_load[0]:+.2f} (t={accel_load[1]:+.2f})")
    if abs(r_4f[1]) > 2:
        print("  => Alpha SURVIVES the acceleration control: there is real selection skill")
        print("     beyond the acceleration premium itself.")
    elif abs(r_4f[1]) < 1.65:
        print("  => Alpha COLLAPSES once you control for the acceleration premium: the Edge is")
        print("     essentially a leveraged acceleration-factor bet, not selection skill.")
    else:
        print("  => Alpha is MARGINAL after the acceleration control: mostly factor harvest,")
        print("     with at most weak residual selection.")


if __name__ == "__main__":
    main()
