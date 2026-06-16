"""edge_neutralize_survivorship.py

Rigorous assessment of (1) SURVIVORSHIP BIAS and (2) OVERFIT / TREND-FITTING
for the FULL Edge spec.

The Edge full spec (from E.EDGE_SPEC):
    signal=accel, hold=42, n=20, mcap_floor=$2B, corr_cap=0.50,
    corr_lookback=126, regime_expo=0.25, cost_bps=10.

Task 1 -- survivorship gap:
    Build the Edge two ways and compare over 1Y/2Y/5Y/MAX:
      (a) FULL pool
      (b) ACTIVE-ONLY: drop names whose meta trading_status != 'Active'
          BEFORE ranking each period.
    The gap is a measurable LOWER BOUND on survivorship inflation. (The truly
    missing long-dead Russell-1000 names -- ~89% of the dead names a real index
    would contain are absent from this pool -- would drag the gap lower still.)

Task 2 -- not-just-a-trend (overfit / regime-fitting):
    (a) first-half vs second-half of the history
    (b) per-calendar-year Edge vs S&P
    (c) pre-2015 vs 2015-onward
    (d) spec-perturbation sensitivity (n=15/25, hold=21/63, cap=0.4/0.6)

Task 3 -- honest haircut + verdict.

Does NOT modify edge_lib.py or qmodel/.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from qmodel import engine

SPEC = dict(E.EDGE_SPEC)
RF_PER = E.RF_PER
PPY = E.PPY


# --------------------------------------------------------------------------- #
# Survivorship: replicate the full-spec selection with an optional active mask #
# --------------------------------------------------------------------------- #
def _status_map():
    data = engine._load_bt_data()
    return {ck: b.get("meta", {}).get("trading_status", "?") for ck, b in data.items()}


def build_edge(pan, active_only=False, status=None, spec=None):
    """Replicate the full-spec Edge selection period by period:
        liquidity floor -> (active mask) -> accel rank -> corr-cap 0.5
        -> regime 25% -> 10bps.
    Returns (gross, net, turn) per-period arrays, aligned to pan.bdates.
    """
    s = {**SPEC, **(spec or {})}
    weights = s["signal"]
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df
        if s["mcap_floor"]:
            d = d[d["pit_mcap"] >= s["mcap_floor"]]
        if active_only:
            d = d[d["company_key"].map(lambda ck: status.get(ck) == "Active")]
        cks = E.corr_cap_select(d, asof=pan.bdates[i], n=s["n"], weights=weights,
                                cap=s["corr_cap"], lookback=s["corr_lookback"]) \
            if s["corr_cap"] else E.select_topN(d, n=s["n"], weights=weights)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
        holds.append(cks)
    gross = E.period_returns(pan, holds)
    if s["regime_expo"] is not None:
        gross = np.where(pan.ma200_on, gross,
                         s["regime_expo"] * gross + (1 - s["regime_expo"]) * RF_PER)
    turn = np.array(turn)
    net = gross - (s["cost_bps"] / 1e4) * turn
    return gross, net, turn


def _win_stats(rets, spxf, k):
    kk = len(rets) if k is None else min(k, len(rets))
    c, dd, sh = E.perf(rets[-kk:])
    cs, _, _ = E.perf(spxf[-kk:])
    return c, sh, dd, cs


# NOTE on annualization: edge_lib.PPY is HARD-CODED to 12.0 (=252/21) and is
# used inside E.perf() to annualize regardless of the actual hold. For the
# 42-day Edge there are only ~6 periods/yr, so E.perf() over-annualizes CAGR by
# ~2x. We therefore ALSO report CAGRs computed with the correct per-hold PPY
# (=252/hold) via cagr_correct(). All RELATIVE numbers (gap vs active-only,
# excess vs S&P) are unaffected because both legs share the same PPY.
def cagr_correct(rets, hold=42):
    r = np.nan_to_num(np.asarray(rets, float))
    if len(r) == 0:
        return float("nan")
    eq = np.cumprod(1 + r); ppy = 252.0 / hold
    return eq[-1] ** (ppy / len(r)) - 1 if eq[-1] > 0 else -1.0


# --------------------------------------------------------------------------- #
def task1_survivorship(pan, status):
    print("=" * 78)
    print("TASK 1 -- SURVIVORSHIP GAP (full pool vs active-only), FULL Edge spec")
    print("=" * 78)
    _, net_full, _ = build_edge(pan, active_only=False)
    _, net_act, _ = build_edge(pan, active_only=True, status=status)
    spx = pan.spxf

    windows = (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None))
    print("\n  [CAGRs below use edge_lib's hard-coded PPY=12 -- inflated ~2x for"
          " the 42d hold; corrected-PPY table follows]")
    print(f"\n  {'window':<6}{'FULL CAGR':>11}{'ACTIVE CAGR':>13}{'GAP(full-act)':>15}"
          f"{'FULL Shrp':>11}{'ACT Shrp':>10}{'S&P CAGR':>10}")
    rows = []
    for wn, k in windows:
        cf, shf, ddf, cs = _win_stats(net_full, spx, k)
        ca, sha, dda, _ = _win_stats(net_act, spx, k)
        gap = cf - ca
        rows.append((wn, cf, ca, gap, shf, sha, cs))
        print(f"  {wn:<6}{cf*100:>10.1f}%{ca*100:>12.1f}%{gap*100:>+14.1f}%"
              f"{shf:>11.2f}{sha:>10.2f}{cs*100:>9.1f}%")

    print(f"\n  CORRECTED ANNUALIZATION (PPY=252/{SPEC['hold']}={252/SPEC['hold']:.1f}):")
    print(f"  {'window':<6}{'FULL CAGR':>11}{'ACTIVE CAGR':>13}{'GAP(full-act)':>15}")
    for wn, k in windows:
        kk = len(net_full) if k is None else min(k, len(net_full))
        cf = cagr_correct(net_full[-kk:], SPEC["hold"])
        ca = cagr_correct(net_act[-kk:], SPEC["hold"])
        print(f"  {wn:<6}{cf*100:>10.1f}%{ca*100:>12.1f}%{(cf-ca)*100:>+14.1f}%")

    # how many inactive names actually get SELECTED in the full-pool basket?
    s = SPEC
    sel_inact = 0
    sel_total = 0
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= s["mcap_floor"]]
        cks = E.corr_cap_select(d, asof=pan.bdates[i], n=s["n"], weights=s["signal"],
                                cap=s["corr_cap"], lookback=s["corr_lookback"])
        sel_total += len(cks)
        sel_inact += sum(1 for ck in cks if status.get(ck) != "Active")
    print(f"\n  Inactive (non-survivor) names actually picked in full-pool basket: "
          f"{sel_inact}/{sel_total} = {100*sel_inact/sel_total:.1f}% of selections")
    print("  NOTE: gap is a LOWER BOUND. ~89% of the dead names a real Russell-1000")
    print("  would contain are MISSING from this pool, so true survivorship drag is larger.")
    return rows


# --------------------------------------------------------------------------- #
def task2_overfit(pan, status):
    print("\n" + "=" * 78)
    print("TASK 2 -- NOT JUST A TREND (overfit / regime-fitting)")
    print("=" * 78)
    _, net, _ = build_edge(pan, active_only=False)
    spx = pan.spxf
    bdates = pan.bdates
    T = len(net)

    # ---- (a) first-half vs second-half -----------------------------------
    print("\n(a) FIRST HALF vs SECOND HALF of history (does it work in both?)")
    h = T // 2
    for lab, sl in (("1st half", slice(0, h)), ("2nd half", slice(h, T))):
        c, dd, sh = E.perf(net[sl])
        cs, _, ss = E.perf(spx[sl])
        d0, d1 = bdates[sl.start].date(), bdates[sl.stop - 1].date()
        print(f"  {lab} {d0}->{d1}: Edge CAGR {c*100:6.1f}%  Sharpe {sh:5.2f}  "
              f"| S&P {cs*100:6.1f}%  excess {(c-cs)*100:+6.1f}%")

    # ---- (b) per-calendar-year -------------------------------------------
    print("\n(b) PER-CALENDAR-YEAR Edge vs S&P (is a few huge years carrying it?)")
    yr = pd.Series(net, index=bdates)
    yrs = pd.Series(spx, index=bdates)
    by = {}
    for y in sorted(set(bdates.year)):
        m = bdates.year == y
        if m.sum() < 2:
            continue
        er = np.prod(1 + np.nan_to_num(net[m])) - 1
        sr = np.prod(1 + np.nan_to_num(spx[m])) - 1
        by[y] = (er, sr, m.sum())
    print(f"  {'year':<6}{'Edge':>9}{'S&P':>9}{'excess':>9}{'n_reb':>7}")
    win_yrs = 0
    for y, (er, sr, n) in by.items():
        beat = er > sr
        win_yrs += beat
        flag = "  <-- big" if er > 0.40 else ""
        print(f"  {y:<6}{er*100:>8.1f}%{sr*100:>8.1f}%{(er-sr)*100:>+8.1f}%{n:>7}{flag}")
    print(f"  Edge beat S&P in {win_yrs}/{len(by)} calendar years "
          f"({100*win_yrs/len(by):.0f}%)")
    # concentration: top-2 years' contribution to total cumulative log return
    er_arr = np.array([v[0] for v in by.values()])
    logc = np.log1p(er_arr)
    order = np.argsort(logc)[::-1]
    top2 = logc[order[:2]].sum() / logc.sum() if logc.sum() != 0 else float("nan")
    print(f"  Top-2 calendar years account for {top2*100:.0f}% of total compounded log return.")

    # ---- (c) pre-2015 vs 2015-onward -------------------------------------
    print("\n(c) PRE-2015 vs 2015-ONWARD")
    pre = bdates.year < 2015
    post = ~pre
    for lab, m in (("pre-2015 ", pre), ("2015+    ", post)):
        c, dd, sh = E.perf(net[m])
        cs, _, _ = E.perf(spx[m])
        print(f"  {lab}: Edge CAGR {c*100:6.1f}%  Sharpe {sh:5.2f}  "
              f"| S&P {cs*100:6.1f}%  excess {(c-cs)*100:+6.1f}%  (n={m.sum()})")

    # ---- (d) spec-perturbation sensitivity -------------------------------
    print("\n(d) SPEC-PERTURBATION SENSITIVITY (robust=stable, fragile=knife-edge)")
    perts = [
        ("BASE n=20,hold=42,cap=0.50", {}),
        ("n=15", {"n": 15}),
        ("n=25", {"n": 25}),
        ("hold=21", {"hold": 21}),
        ("hold=63", {"hold": 63}),
        ("cap=0.40", {"corr_cap": 0.40}),
        ("cap=0.60", {"corr_cap": 0.60}),
    ]
    print("  (CAGR uses CORRECT per-hold annualization PPY=252/hold, so hold=21/42/63"
          " are comparable)")
    print(f"  {'variant':<28}{'MAX CAGR':>10}{'maxDD':>8}{'vs S&P':>9}")
    cagrs = []
    for lab, ov in perts:
        hold = ov.get("hold", SPEC["hold"])
        # hold changes the panel entirely -> reload appropriate panel
        if hold != SPEC["hold"]:
            p = E.load_edge_panel(hold=hold)
        else:
            p = pan
        _, net_v, _ = build_edge(p, active_only=False, spec=ov)
        _, dd, _ = E.perf(net_v)
        c = cagr_correct(net_v, hold)
        cs = cagr_correct(p.spxf, hold)
        cagrs.append(c)
        print(f"  {lab:<28}{c*100:>9.1f}%{dd*100:>7.0f}%{(c-cs)*100:>+8.1f}%")
    spread = max(cagrs) - min(cagrs)
    print(f"  CAGR spread across all perturbations: {spread*100:.1f} pts "
          f"(min {min(cagrs)*100:.1f}% .. max {max(cagrs)*100:.1f}%)")
    print("  Robust if spread is small relative to the ~16% base; fragile if a single"
          " knob swings it.")


# --------------------------------------------------------------------------- #
def main():
    pan = E.load_edge_panel(hold=SPEC["hold"])
    status = _status_map()
    print(f"Edge panel: {pan.T} rebalances, {pan.bdates[0].date()} -> "
          f"{pan.bdates[-1].date()}, market>200dMA {pan.ma200_on.mean()*100:.0f}% of time")
    print(f"Pool: 1000 Active + 111 Inactive (88 inactive ever appear in panels)\n")
    rows = task1_survivorship(pan, status)
    task2_overfit(pan, status)


if __name__ == "__main__":
    main()
