"""Rigor check on the GP/assets quality-gate overlay (the one fundamental win).

Stage-2 + robustness found: gating out the lowest-gross-profitability ~40% of the
pool before the accel pick gives a smooth Sharpe plateau (peak ~gate 0.40), holds
in both halves, and shallower drawdowns. This applies the campaign's overfitting-
adjusted bar (edge_rigor) to the GATED single-sleeve period returns vs baseline:
  * Newey-West t on the excess vs S&P (is the gated book still significant?)
  * Deflated Sharpe Ratio with the trial count inflated for THIS campaign.
Trial Sharpes come from the actual gate sweep we ran (honest dispersion).

Run: .venv/Scripts/python.exe edge_test_fund_rigor.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np
import edge_lib as E
from edge_rigor import newey_west_t, deflated_sharpe
from edge_test_fund_stage2 import _select_overlay
from edge_test_fund_canonical import attach as attach_canon

S = E.EDGE_SPEC
PPY = 252.0 / S["hold"]

# Annual Sharpes actually observed across the gp_assets gate sweep (n=7 & n=10,
# full + both halves) -> the campaign's trial dispersion for the deflation.
TRIAL_SR_ANNUAL = [0.94, 0.87, 0.88, 0.90, 0.95, 0.96, 1.01, 0.98,
                   0.90, 0.88, 0.86, 0.92, 0.94, 0.95, 1.00, 1.01,
                   0.82, 0.79, 0.78, 0.89, 0.89, 0.89]  # + tilts
N_TRIALS = 150 + 70                                      # price campaign + fundamental campaign


def gated_periods(fund, n, mode, param):
    """Single-sleeve (offset 0) per-period NET returns + aligned S&P, gp_assets gate."""
    pan = E.load_edge_panel(hold=S["hold"], offset_days=0)
    panels_f, _ = attach_canon(pan, fund)
    holds, turn = _select_overlay(pan, panels_f, n, "gp_assets", 1, mode, param)
    gross = E.period_returns(pan, holds)
    gross = np.where(pan.ma200_on, gross,
                     S["regime_expo"] * gross + (1 - S["regime_expo"]) * pan.rf_per)
    net = gross - (S["cost_bps"] / 1e4) * np.asarray(turn)
    return np.asarray(net, float), np.asarray(pan.spxf, float)


def report(name, net, spx):
    excess = net - spx
    t_raw = excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
    t_nw, T, L = newey_west_t(excess)
    var_sr = float(np.var(np.array(TRIAL_SR_ANNUAL) / np.sqrt(PPY)))
    dsr, sr, sr_star, g3, g4, _ = deflated_sharpe(net, N_TRIALS, var_sr)
    print(f"\n--- {name} ---")
    print(f"  excess vs S&P ~{excess.mean()*PPY*100:+.1f}%/yr | NW t(L={L})={t_nw:+.2f} "
          f"({'sig' if abs(t_nw) > 2 else 'NOT sig'}) | raw t={t_raw:+.2f}")
    print(f"  per-period SR {sr:.3f} (~ann {sr*np.sqrt(PPY):.2f}); skew {g3:+.2f} kurt {g4:.1f}; "
          f"SR* {sr_star:.3f}")
    print(f"  DSR = {dsr:.3f}  ({'PASSES' if dsr > 0.95 else 'FAILS'} >0.95 bar, {N_TRIALS} trials)")


def main():
    with open("data/cache/fund_canonical.pkl", "rb") as f:
        fund = pickle.load(f)
    print("=" * 74)
    print("RIGOR — GP/assets quality gate vs baseline (single-sleeve period returns)")
    print("=" * 74)
    for n in (7, 10):
        print(f"\n================== n={n} ==================")
        nb, sb = gated_periods(fund, n, "baseline", 0.0)
        report(f"baseline (accel only), n={n}", nb, sb)
        ng, sg = gated_periods(fund, n, "gate", 0.40)
        report(f"gp_assets gate 0.40, n={n}", ng, sg)


if __name__ == "__main__":
    main()
