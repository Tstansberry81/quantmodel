"""Drawdown reduction, re-measured on the SHIPPED monthly grid.

Everything previously known about drawdown here was measured on the old 42-day
grid at its one lucky phase -- the phase sweep showed that anchor was the best of
six (-30.8% vs a -44.2% median). So the vol-target response curve, the regime
numbers, all of it, needs re-measuring before any of it guides a decision.

Every variant below changes only OVERLAY parameters, so they all reuse the one
cached panel and the selection -- cheap. Variants that change selection (n,
sector_cap) are marked and cost more.
"""
import numpy as np
import edge_lib as E

S = E.EDGE_SPEC
SIG = tuple(sorted(S["signal"].items()))

def run(**over):
    s = {**S, **over}
    idx, model, spx, ndx, to, _ = E._edge_daily(
        s["hold"], s["n"], s["mcap_floor"], s["corr_cap"], s["corr_lookback"],
        s["regime_expo"], s["cost_bps"], SIG, 0.0, s["growth_thresh"], False,
        s["continuous_regime"], False, s["sector_cap"], s.get("name_trend", False),
        s["vol_target"], s["vol_lookback"], s["vol_cap"], s["rebal_months"])
    c, dd, sh = E._perf_daily(model)
    so = E._sortino_daily(model)
    eq = np.cumprod(1 + np.nan_to_num(model))
    d = eq / np.maximum.accumulate(eq) - 1
    # time spent more than 20% underwater: a drawdown you sit in for years is a
    # different product from one you exit in months, and maxDD alone hides that.
    pain = float((d < -0.20).mean())
    mar = c / abs(dd) if dd else float("nan")
    return c, sh, so, dd, mar, pain, to

CASES = [
    ("SHIPPED (vol 25%, 21d)",        {}),
    ("vol target 20%",                dict(vol_target=0.20)),
    ("vol target 15%",                dict(vol_target=0.15)),
    ("vol target 12%",                dict(vol_target=0.12)),
    ("vol 15% + 42d lookback",        dict(vol_target=0.15, vol_lookback=42)),
    ("vol 15% + 63d lookback",        dict(vol_target=0.15, vol_lookback=63)),
    ("vol 20% + 42d lookback",        dict(vol_target=0.20, vol_lookback=42)),
    ("regime 0% (full cash below MA)", dict(regime_expo=0.0)),
    ("regime 10%",                    dict(regime_expo=0.10)),
    ("vol 15% + regime 0%",           dict(vol_target=0.15, regime_expo=0.0)),
    ("no vol target (regime only)",   dict(vol_target=None)),
]
print(f"{'variant':<32}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>9}"
      f"{'MAR':>6}{'%<-20%':>8}{'turn':>7}")
for label, over in CASES:
    c, sh, so, dd, mar, pain, to = run(**over)
    print(f"{label:<32}{c*100:7.2f}%{sh:8.3f}{so:9.3f}{dd*100:8.2f}%"
          f"{mar:6.2f}{pain*100:7.1f}%{to:7.3f}")
