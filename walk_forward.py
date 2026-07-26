"""Out-of-sample test: would a 2012-era version of this analysis have worked?

signal_scan.py ranked signals using the WHOLE sample, and config_test.py then
confirmed that 12-1 momentum beats the S&P over that same sample. That is not
evidence yet -- the signal was chosen with knowledge of the period it is being
judged on. The honest question is:

    Using ONLY the first half of history, which signal would we have picked?
    And how did that choice actually do over the second half, untouched?

So: rank every candidate by its top-N tail t-stat on the IN-SAMPLE half, take
the winner, and evaluate its portfolio -- full spec, corr cap, growth gate,
regime overlay, costs -- over the OUT-OF-SAMPLE half only. The shipped accel
signal is carried through as the control.

Selection sees nothing past the split date. If the winner also wins out of
sample, the effect is more likely real than fitted; if it collapses, we found a
period artifact and should say so.

Run:  .venv-mac/bin/python walk_forward.py
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")

import numpy as np
from scipy import stats

import edge_data as D
import edge_lib as E

HOLD, TOP_N, MCAP_FLOOR = 42, 10, 2e9
CORR_CAP, CORR_LOOKBACK = 0.50, 126
REGIME_EXPO, COST_BPS = 0.25, 10.0
GROWTH_MIX, GROWTH_THRESH = 0.75, 0.15

CANDIDATES = {
    "accel": +1, "ret_63": +1, "ret_126": +1, "ret_12_1": +1,
    "hi_252": +1, "rs_63": +1, "ret_21": -1, "vol_21": -1, "rev_growth": +1,
}


def tail_t(pan, col, sign, lo, hi):
    """Top-N-minus-universe forward return t-stat over rebalances [lo, hi)."""
    tails = []
    for i in range(lo, hi):
        d = pan.panels[i]
        d = d[d["pit_mcap"] >= MCAP_FLOOR].dropna(subset=["fwd_ret", col])
        d = d[np.isfinite(d["fwd_ret"]) & np.isfinite(d[col])]
        if len(d) < 100:
            continue
        fwd = d["fwd_ret"].astype(float)
        srt = fwd.iloc[np.argsort((sign * d[col].astype(float)).values)]
        tails.append(float(srt.tail(TOP_N).mean() - fwd.mean()))
    a = np.array(tails)
    t = float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))) if len(a) > 2 and a.std(ddof=1) > 0 else 0.0
    return float(a.mean()), t


def portfolio(pan, signal: dict, lo: int, hi: int, *, sector_neutral=False,
              beta_neutral=False, corr_mode="max", cap=CORR_CAP, weight="equal"):
    """Net per-rebalance returns for the full spec over [lo, hi)."""
    holds, turn, prev = [], [], None
    for i in range(lo, hi):
        d = pan.panels[i][pan.panels[i]["pit_mcap"] >= MCAP_FLOOR]
        cks = E._blend_select(d, pan.bdates[i], TOP_N, signal, cap,
                              CORR_LOOKBACK, GROWTH_MIX, GROWTH_THRESH,
                              corr_mode=corr_mode, sector_neutral=sector_neutral,
                              beta_neutral=beta_neutral)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
        holds.append(cks)
    # period_returns walks the whole panel, so slice its output to our range
    gross_all = E.period_returns(pan, [[]] * lo + holds + [[]] * (pan.T - hi),
                                 weight=weight)
    gross = gross_all[lo:hi]
    on = pan.ma200_on[lo:hi]
    gross = np.where(on, gross, REGIME_EXPO * gross + (1 - REGIME_EXPO) * pan.rf_per)
    return gross - (COST_BPS / 1e4) * np.array(turn)


def vol_scaled(net, ppy, rf_per, target=0.15, lookback=6, cap=1.0):
    """Barroso-Santa-Clara volatility scaling, long-only.

    Momentum's pathology is not weak average return, it's crashes: the OOS book
    earns more than the index while drawing down far more. Scaling exposure by
    target / trailing-realised-vol de-risks exactly when the strategy itself is
    turbulent. STRICTLY CAUSAL -- rebalance i only ever sees returns < i -- and
    capped at 1.0, i.e. de-risking only, never leverage (this is a long-only
    cash account, not a margin book). Idle weight earns the risk-free rate."""
    out = np.asarray(net, float)
    scaled = np.empty_like(out)
    for i in range(len(out)):
        past = out[max(0, i - lookback):i]
        if len(past) < 3 or past.std(ddof=1) <= 0:
            e = 1.0
        else:
            e = min(cap, target / (past.std(ddof=1) * np.sqrt(ppy)))
        scaled[i] = e * out[i] + (1 - e) * rf_per
    return scaled


def main() -> int:
    pan = E.load_edge_panel(hold=HOLD)
    split = pan.T // 2
    ppy = pan.ppy
    print(f"panel: {D.meta().get('source')} | {pan.T} rebalances "
          f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()}")
    print(f"IN-SAMPLE  : {pan.bdates[0].date()} -> {pan.bdates[split-1].date()} "
          f"({split} rebalances)  <- selection sees ONLY this")
    print(f"OUT-SAMPLE : {pan.bdates[split].date()} -> {pan.bdates[-1].date()} "
          f"({pan.T - split} rebalances)\n")

    print(f"{'candidate':<12}{'IS tail':>10}{'IS t':>7}   {'OOS tail':>10}{'OOS t':>7}")
    print("-" * 48)
    ranked = []
    for col, sign in CANDIDATES.items():
        m_is, t_is = tail_t(pan, col, sign, 0, split)
        m_oos, t_oos = tail_t(pan, col, sign, split, pan.T)
        ranked.append((t_is, col, sign, m_is, m_oos, t_oos))
    for t_is, col, sign, m_is, m_oos, t_oos in sorted(ranked, reverse=True):
        print(f"{col:<12}{m_is*100:>+9.2f}%{t_is:>7.1f}   {m_oos*100:>+9.2f}%{t_oos:>7.1f}")

    best_t, best, best_sign = max(ranked)[0], max(ranked)[1], max(ranked)[2]
    print(f"\n-> In-sample selection picks: {best} (t={best_t:.1f}). "
          f"Judging it out of sample only.\n")

    sp_oos = pan.spxf[split:pan.T]
    sp_cagr, sp_dd, sp_sh = E.perf(sp_oos, ppy)
    print(f"{'OUT-OF-SAMPLE':<26}{'CAGR':>9}{'excess':>10}{'Sharpe':>9}{'maxDD':>9}")
    print("-" * 63)
    print(f"{'S&P 500 (benchmark)':<26}{sp_cagr*100:>8.1f}%{'--':>10}{sp_sh:>9.2f}{sp_dd*100:>8.1f}%")
    picked_net = None
    for label, sig in ((f"picked: {best}", {best: 1.0}),
                       ("shipped: accel", {"accel": 1.0})):
        net = portfolio(pan, sig, split, pan.T)
        if picked_net is None:
            picked_net = net
        c, dd, sh = E.perf(net, ppy)
        print(f"{label:<26}{c*100:>8.1f}%{(c-sp_cagr)*100:>+9.1f}%{sh:>9.2f}{dd*100:>8.1f}%")

    # The OOS failure is risk, not return: more CAGR than the index but a far
    # deeper drawdown. Vol-scaling targets exactly that, so test it here rather
    # than as another full-sample config.
    vs = vol_scaled(picked_net, ppy, pan.rf_per)
    c, dd, sh = E.perf(vs, ppy)
    print(f"{'  + vol-scaled (15% tgt)':<26}{c*100:>8.1f}%{(c-sp_cagr)*100:>+9.1f}%"
          f"{sh:>9.2f}{dd*100:>8.1f}%")

    # --- portfolio-construction variants, stated BEFORE running -------------
    # The signal stays fixed at the pre-2012 pick, so these test construction
    # only. Summative correlation uses a TIGHTER cap on purpose: mean|rho| is
    # always <= max|rho|, so reusing 0.50 would LOOSEN the constraint rather
    # than tighten it (measured: it admitted a 0.55 pair and raised the
    # basket's average correlation from 0.274 to 0.323).
    sig = {best: 1.0}
    variants = [
        ("+ sector+beta neutral", dict(sector_neutral=True, beta_neutral=True)),
        ("+ summative corr .35",  dict(corr_mode="mean", cap=0.35)),
        ("+ inverse-vol weights", dict(weight="invvol")),
        ("= all three",           dict(sector_neutral=True, beta_neutral=True,
                                       corr_mode="mean", cap=0.35, weight="invvol")),
    ]
    for label, kw in variants:
        net = portfolio(pan, sig, split, pan.T, **kw)
        c, dd, sh = E.perf(net, ppy)
        print(f"{label:<26}{c*100:>8.1f}%{(c-sp_cagr)*100:>+9.1f}%{sh:>9.2f}{dd*100:>8.1f}%")

    print("\nSelection used only pre-split data. Data: Sharadar SEP/SF1 "
          "point-in-time (ART, datekey), delisted INCLUDED; universe top-1000 "
          "by PIT market cap >= $2B; filing lag +1 trading day; rebalance 42 "
          "trading days; net of 10bps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
