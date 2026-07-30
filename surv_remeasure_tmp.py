"""Re-measure residual survivorship exposure on v5.

The site says "about 0.2 percentage points of CAGR". That came from the
2026-07-29 DELIST_HAIRCUT change (None -> 0.0) measured on v3 — a different
signal grid AND before the solvency screens. Two model versions stale.

THE MEASUREMENT. DELIST_HAIRCUT=None EXCLUDES any name whose price series ends
inside the holding window; 0.0 includes it, truncated at its last trade. The
None variant is LOOK-AHEAD: it drops a name on date d because it knows the
company stops trading within the next month, which a trader on d cannot know.
The gap between the two is what that look-ahead was worth — i.e. the residual
survivorship exposure.

DELIST_HAIRCUT is a module global, not a function argument, so it is NOT in any
cache key beyond the panel fingerprint: reset_caches() after mutating it or the
second run silently returns the first one's numbers (RESEARCH_RULES #5).
"""
import edge_lib as E

def run(haircut, label):
    E.DELIST_HAIRCUT = haircut
    E.reset_caches()                      # REQUIRED — see docstring
    bt = E.run_edge_backtest("MAX")
    m = bt["performance"]["model"]
    print(f"{label:<44}{m['cagr']*100:7.2f}%{m['sharpe']:8.3f}{m['max_drawdown']*100:8.2f}%")
    return m

print(f"{'variant':<44}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}")
shipped = run(0.0,  "SHIPPED: dying names held to last trade")
looky   = run(None, "LOOK-AHEAD: dying names excluded")
print()
print(f"residual survivorship exposure = {(looky['cagr']-shipped['cagr'])*100:.2f} pts of CAGR")
print(f"  (site currently claims ~0.2 pts, measured on v3)")
E.DELIST_HAIRCUT = 0.0; E.reset_caches()   # restore
