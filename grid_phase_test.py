"""Does the REBALANCE GRID's start date carry the result? (Yes. See below.)

WHY THIS EXISTS
---------------
The grid used to be `monthly[:-hold][::hold]` -- a fixed stride of `hold` trading
days from whatever day sat 400 calendar days into the sample. That start date was
never chosen; it fell out of a lookback buffer. So it was never swept, and a
number that came out of it went to the website.

RESULT (2026-07-30, shipped spec, MAX window, production daily basis):

    phase   CAGR    Sharpe   maxDD     worst drawdown
      0    18.29%   0.934   -30.76%    2021-08 -> 2022-01
      7    16.95%   0.874   -43.62%    2021-08 -> 2023-10
     14    17.08%   0.884   -43.21%    2021-02 -> 2023-10
     21    15.79%   0.826   -44.79%    2021-08 -> 2023-10
     28    15.68%   0.824   -47.79%    2021-02 -> 2023-10
     35    16.33%   0.850   -48.15%    2021-06 -> 2023-10

Phase 0 -- the shipped one -- is the best of six on every axis, by a lot. The
drawdown spread is 17.4 points and the median is -44.21%. The last column is the
tell: every other phase stays underwater until October 2023, while phase 0
recovers in January 2022. It happened to exit the November-2021 momentum peak on
a lucky week. See RESEARCH_RULES #8.

The fix was not to find a better phase. It was to stop having one: the grid is
now anchored to calendar month starts, so its phase is set by the calendar
instead of by an accident.

Run:
    .venv-mac/bin/python grid_phase_test.py clocks   # compare grid conventions
    .venv-mac/bin/python grid_phase_test.py phase    # the sweep above
    .venv-mac/bin/python grid_phase_test.py diag     # coverage + episode detail

NOTE each variant builds its own panel (minutes, and ~300-600MB each on the full
research artifact). This is not a quick script.
"""
from __future__ import annotations
import sys

import numpy as np

import edge_lib as E

S = E.EDGE_SPEC
SIG = tuple(sorted(S["signal"].items()))


def _daily(hold, rebal_months, offset_days=0):
    """The shipped spec's daily net curve on a chosen grid.

    Goes through the selection layer rather than run_edge_backtest because
    `offset_days` is the whole point here and _edge_daily does not accept one --
    it derives offsets from `stagger`. The first version of this swept a variable
    that function ignores and returned six byte-identical rows (RESEARCH_RULES
    #6b), which is why the phase sweep below self-checks against a known number.
    """
    M = E.D.daily_return_matrix()
    rd = E._ma200_daily_state() if (S["continuous_regime"] and S["regime_expo"] is not None) else None
    pan, holds, turn = E._select_holds_cached(
        hold, offset_days, S["n"], SIG, S["mcap_floor"], S["corr_cap"],
        S["corr_lookback"], 0.0, S["growth_thresh"], False, S["sector_cap"],
        False, rebal_months)
    model = E._sleeve_daily(pan, holds, turn, M, hold, S["regime_expo"],
                            S["cost_bps"], rd).dropna()
    if S["vol_target"]:
        model = E._apply_vol_target(model, S["vol_target"], S["vol_lookback"],
                                    S["vol_cap"], S["cost_bps"])
    return model, pan


def _worst(idx, r):
    eq = np.cumprod(1 + np.nan_to_num(r))
    dd = eq / np.maximum.accumulate(eq) - 1
    t = int(np.argmin(dd)); p = int(np.argmax(eq[:t + 1])) if t else 0
    return idx[p].date(), idx[t].date()


def _episodes(idx, r, k=4):
    """The k deepest NON-OVERLAPPING drawdowns. One headline maxDD cannot tell you
    whether a grid change deepened the same episode or found a different one."""
    eq = np.cumprod(1 + np.nan_to_num(r))
    dd = eq / np.maximum.accumulate(eq) - 1
    out, used = [], np.zeros(len(dd), bool)
    for _ in range(k):
        d = np.where(used, 0.0, dd)
        t = int(np.argmin(d))
        if d[t] > -1e-9:
            break
        p = int(np.argmax(eq[:t + 1])) if t else 0
        e = t
        while e + 1 < len(eq) and eq[e + 1] < eq[p]:
            e += 1
        out.append((idx[p].date(), idx[t].date(), float(dd[t])))
        used[p:e + 1] = True
    return out


def clocks() -> None:
    """Compare grid CONVENTIONS at a fixed spec."""
    cases = [("legacy stride h42 (pre-2026-07-30)", 42, None),
             ("month-anchored, 2-month  h42", 42, 2),
             ("month-anchored, MONTHLY  h21", 21, 1),
             ("legacy stride h21", 21, None)]
    print(f"{'variant':<36}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}{'turn':>8}  live book")
    for label, hold, rm in cases:
        model, pan = _daily(hold, rm)
        c, dd, sh = E._perf_daily(model.to_numpy())
        _, _, turn = E._select_holds_cached(
            hold, 0, S["n"], SIG, S["mcap_floor"], S["corr_cap"], S["corr_lookback"],
            0.0, S["growth_thresh"], False, S["sector_cap"], False, rm)
        print(f"{label:<36}{c*100:7.2f}%{sh:8.3f}{dd*100:8.2f}%{float(np.mean(turn)):8.3f}"
              f"  {pan.live_date.date()} ({pan.T} rebals)")


def phase() -> None:
    """Slide the LEGACY grid's anchor. The headline result of this file."""
    print(f"{'phase':>6} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>9}   worst drawdown")
    dds = []
    for off in (0, 7, 14, 21, 28, 35):
        model, pan = _daily(42, None, off)
        r = model.to_numpy()
        c, dd, sh = E._perf_daily(r)
        dds.append(dd)
        p, t = _worst(model.index, r)
        print(f"{off:>6} {c*100:7.2f}% {sh:7.3f} {dd*100:8.2f}%   {p} -> {t}")
        if off == 0 and abs(dd + 0.3076) > 0.002:
            print("  !! off=0 did not reproduce the known -30.76%; harness is wrong, STOP")
            return
    print(f"\nmaxDD across phases: best {max(dds)*100:.2f}%  worst {min(dds)*100:.2f}%  "
          f"median {np.median(dds)*100:.2f}%  spread {(max(dds)-min(dds))*100:.1f}pts")
    print("The shipped phase was the BEST of six. See RESEARCH_RULES #8.")


def diag() -> None:
    """Curve COVERAGE plus per-episode drawdowns.

    Coverage matters because holding windows are derived from the next rebalance
    date: if that arithmetic left holes, `.dropna()` would delete real market days
    and _perf_daily -- which annualizes by 252/len(r) -- would quietly report a
    HIGHER CAGR for a worse curve."""
    cal = E.D.daily_return_matrix().index
    for label, hold, rm in (("legacy stride h42", 42, None),
                            ("month-anchored 2M", 42, 2),
                            ("month-anchored 1M (SHIPPED)", 21, 1)):
        model, pan = _daily(hold, rm)
        idx = model.index
        span = cal[(cal >= idx[0]) & (cal <= idx[-1])]
        gaps = len(span) - len(idx)
        c, dd, sh = E._perf_daily(model.to_numpy())
        print(f"\n=== {label} ===")
        print(f"  {pan.T} rebalances · live book {pan.live_date.date()} · "
              f"{idx[0].date()} -> {idx[-1].date()}")
        print(f"  curve days {len(idx)} of {len(span)} in span -> {gaps} missing  "
              f"{'OK (gap-free)' if gaps == 0 else '<-- HOLES, numbers are wrong'}")
        print(f"  CAGR {c*100:.2f}%  Sharpe {sh:.3f}  maxDD {dd*100:.2f}%")
        for p, t, v in _episodes(idx, model.to_numpy()):
            print(f"    {v*100:7.2f}%   {p} -> {t}")


if __name__ == "__main__":
    part = (sys.argv[1] if len(sys.argv) > 1 else "phase").lower()
    {"clocks": clocks, "phase": phase, "diag": diag}.get(part, phase)()
