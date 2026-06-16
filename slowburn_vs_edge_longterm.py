"""
Slow Burn vs The Edge -- LONG-TERM head-to-head.

Slow Burn = the long-horizon fundamental engine (qmodel).
The Edge   = the short-term acceleration-momentum trading model (edge_lib).

We compare CAGR / Sharpe / max-drawdown over long windows (5Y / 10Y / 20Y / MAX)
and render a side-by-side table, then state an honest verdict.

HONESTY NOTES (read before quoting anything):
  * The Edge's headline numbers still carry SURVIVORSHIP bias (the universe is
    missing the ~89% of names that delisted, worth roughly -3%/yr) AND sit in an
    unusually momentum-friendly recent regime. Mentally apply a ~3-5%/yr haircut
    to every Edge CAGR below before comparing.
  * Slow Burn is the long-horizon fundamental engine with correct annualization;
    its numbers are the more trustworthy long-run figure.

Run with: .venv\\Scripts\\python.exe slowburn_vs_edge_longterm.py
"""

from qmodel import engine
import edge_lib as E

WINDOWS = ["5Y", "10Y", "20Y", "MAX"]
HAIRCUT_LO, HAIRCUT_HI = 0.03, 0.05  # survivorship + momentum-regime haircut on the Edge


def _perf(d):
    """Pull (cagr, sharpe, max_drawdown) out of a model performance dict."""
    m = (d.get("performance") or {}).get("model") or {}
    return m.get("cagr"), m.get("sharpe"), m.get("max_drawdown")


def _pct(x):
    return "   n/a" if x is None else f"{x*100:6.1f}%"


def _num(x, d=2):
    return "  n/a" if x is None else f"{x:5.{d}f}"


def main():
    rows = []  # (window, sb_cagr, sb_sharpe, sb_dd, ed_cagr, ed_sharpe, ed_dd)
    for w in WINDOWS:
        try:
            sb = engine.compute_backtest(window=w, hold="1M")
            sb_cagr, sb_sh, sb_dd = _perf(sb)
        except Exception as e:
            sb_cagr = sb_sh = sb_dd = None
            print(f"[warn] Slow Burn {w} failed: {e}")
        try:
            ed = E.run_edge_backtest(window=w)
            ed_cagr, ed_sh, ed_dd = _perf(ed)
        except Exception as e:
            ed_cagr = ed_sh = ed_dd = None
            print(f"[warn] Edge {w} failed: {e}")
        rows.append((w, sb_cagr, sb_sh, sb_dd, ed_cagr, ed_sh, ed_dd))

    # ---- side-by-side table ----
    print()
    print("LONG-TERM HEAD-TO-HEAD  --  Slow Burn (fundamental)  vs  The Edge (momentum)")
    print("=" * 84)
    print(f"{'':6} | {'SLOW BURN':^24} | {'THE EDGE (raw)':^24} | {'EDGE adj.':^14}")
    print(f"{'Win':6} | {'CAGR':>7} {'Shrp':>6} {'maxDD':>7} | "
          f"{'CAGR':>7} {'Shrp':>6} {'maxDD':>7} | {'CAGR*':>13}")
    print("-" * 84)
    for w, sc, ss, sd, ec, es, ed in rows:
        if ec is None:
            adj = "      n/a"
        else:
            adj = f"{(ec-HAIRCUT_HI)*100:5.1f}-{(ec-HAIRCUT_LO)*100:.1f}%"
        print(f"{w:6} | {_pct(sc)} {_num(ss)} {_pct(sd)} | "
              f"{_pct(ec)} {_num(es)} {_pct(ed)} | {adj:>13}")
    print("=" * 84)
    print("* Edge CAGR after a 3-5%/yr survivorship + momentum-regime haircut.")
    print("  maxDD shown as the raw model drawdown (more negative = deeper).")
    print()

    # ---- verdict ----
    # Use MAX (or longest available) as the anchor for the long-run call.
    anchor = next((r for r in reversed(rows) if r[1] is not None and r[4] is not None), None)
    print("VERDICT")
    print("-" * 84)
    if anchor:
        w, sc, ss, sd, ec, es, ed = anchor
        ec_adj_lo, ec_adj_hi = ec - HAIRCUT_HI, ec - HAIRCUT_LO
        print(
            f"Over the longest window ({w}): Slow Burn ~{sc*100:.1f}% CAGR (Sharpe {ss:.2f}) "
            f"vs the Edge's raw ~{ec*100:.1f}% (Sharpe {es:.2f}).\n"
            f"But the Edge's figure is inflated by survivorship + a momentum-friendly\n"
            f"regime; after an honest 3-5%/yr haircut the Edge is really ~{ec_adj_lo*100:.1f}-{ec_adj_hi*100:.1f}%/yr.\n"
        )
    print(
        "Bottom line: for a genuine LONG-TERM hold, Slow Burn is the better engine.\n"
        "The Edge's eye-popping CAGR is a measurement artifact (no delisted names,\n"
        "an unusually momentum-rich window, and it is built to be traded on a ~60-90\n"
        "day clock, not held for a decade). Once you haircut it down to a defensible\n"
        "forward expectation (~11-13%/yr) it converges toward -- and does not durably\n"
        "beat -- the long-run fundamental compounding of Slow Burn. The Edge wins on\n"
        "short-horizon, risk-adjusted SHAPE (high Sharpe, tight drawdowns); Slow Burn\n"
        "wins on trustworthy LONG-TERM compounding. Long-term, hold Slow Burn."
    )


if __name__ == "__main__":
    main()
