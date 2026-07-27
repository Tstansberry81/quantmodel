"""Sector cap: how much return does bounding concentration actually cost?

Five candidates fixed in advance (no cap, 2, 3, 4, 5 names per sector). A cap
is a RISK control justified before any backtest -- the live book was 9/10
Technology, so one industry shock owns the portfolio -- so this is choosing a
setting for a control we want on principle, not fishing for a better number.
Chosen on drawdown reduction per point of return given up.

Production basis (run_edge_backtest, daily curve), per RESEARCH_RULES.md #1.
"""
import os
os.environ.setdefault("EDGE_USE_PIT_UNIVERSE", "0")
import edge_lib as E

print(f"{'sector cap':<14}{'CAGR':>8}{'excess':>9}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}"
      f"{'DD saved':>10}{'CAGR cost':>11}{'ratio':>8}")
print("-" * 85)
base = None
for cap in (None, 5, 4, 3, 2):
    bt = E.run_edge_backtest(window="MAX", spec={**E.EDGE_SPEC, "sector_cap": cap})
    m, s = bt["performance"]["model"], bt["performance"]["sp500"]
    if base is None:
        base = (m["cagr"], m["max_drawdown"])
    dd_saved = (m["max_drawdown"] - base[1]) * 100      # positive = shallower
    cagr_cost = (base[0] - m["cagr"]) * 100             # positive = gave up
    ratio = (dd_saved / cagr_cost) if cagr_cost > 0.05 else float("nan")
    print(f"{str(cap or 'none'):<14}{m['cagr']*100:>7.1f}%{(m['cagr']-s['cagr'])*100:>+8.1f}%"
          f"{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{m['max_drawdown']*100:>7.0f}%"
          f"{dd_saved:>9.1f}p{cagr_cost:>10.1f}p{ratio:>8.2f}")
print("\nratio = drawdown points saved per point of CAGR given up (higher is better).")
print(f"S&P benchmark: Sharpe {s['sharpe']:.2f}, maxDD {s['max_drawdown']*100:.0f}%")
