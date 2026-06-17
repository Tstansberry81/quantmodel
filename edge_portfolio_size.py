"""Exploration (not wired into the site):

1) Portfolio-size sweep — how does basket size n (10..20) affect the full-spec
   Edge (mix=0.75, hold=42, $2B floor, corr-cap 0.5, regime 25%, 10bps)?
2) Feasibility probe — can we attach, per paper-trading-log entry, the actual
   stocks the model selected that rebalance (a "Portfolio" dropdown)? The holds
   are already computed inside _edge_full; here we reconstruct them + each name's
   own forward return to show the data exists and is cheap.

Run:  .venv/Scripts/python.exe edge_portfolio_size.py
"""
from __future__ import annotations
import numpy as np

import edge_lib as E
from qmodel import engine

SPEC = E.EDGE_SPEC
HOLD = SPEC["hold"]            # 42
MIX = SPEC["growth_mix"]      # 0.75


def _holds_for(pan, n):
    """Recompute the per-rebalance baskets for basket size n (same logic as
    _edge_full, but returning the holdings instead of throwing them away)."""
    weights = dict(SPEC["signal"])
    holds = []
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= SPEC["mcap_floor"]] if SPEC["mcap_floor"] else df
        cks = E._blend_select(d, pan.bdates[i], n, weights, SPEC["corr_cap"],
                              SPEC["corr_lookback"], MIX, SPEC["growth_thresh"])
        holds.append(cks)
    return holds


def sweep():
    print("Building Edge panel (one-time)…")
    pan = E.load_edge_panel(hold=HOLD)
    ppy = pan.ppy
    T = len(pan.spxf)
    win = {"1Y": int(round(ppy)), "5Y": int(round(5 * ppy)), "MAX": T}

    sp_max = E.perf(pan.spxf, ppy)[0]
    print(f"\nFull-spec Edge, growth_mix={MIX:.0%}, hold={HOLD}d, "
          f"{T} rebalances. S&P MAX CAGR = {sp_max*100:.1f}%\n")
    hdr = f"{'n':>3} {'MAX CAGR':>9} {'Sharpe':>7} {'maxDD':>7} {'alpha':>7} {'1Y':>7} {'5Y':>7} {'turn/yr':>8}"
    print(hdr); print("-" * len(hdr))
    for n in range(10, 21):
        _, gross, net, turn, spxf, ndxf, _holds = E._edge_full(
            HOLD, n, SPEC["mcap_floor"], SPEC["corr_cap"], SPEC["corr_lookback"],
            SPEC["regime_expo"], SPEC["cost_bps"], tuple(sorted(SPEC["signal"].items())),
            MIX, SPEC["growth_thresh"])
        c, dd, sh = E.perf(net, ppy)
        c1 = E.perf(net[T - win["1Y"]:], ppy)[0]
        c5 = E.perf(net[T - win["5Y"]:], ppy)[0]
        to = float(np.mean(turn)) * ppy
        print(f"{n:>3} {c*100:8.1f}% {sh:7.2f} {dd*100:6.0f}% {(c-sp_max)*100:+6.1f}% "
              f"{c1*100:6.1f}% {c5*100:6.1f}% {to:7.1f}x")
    return pan


def feasibility_probe(pan):
    """Show that each rebalance's basket + per-name return is recoverable."""
    print("\n\n=== FEASIBILITY: per-rebalance holdings (for the log dropdown) ===")
    data = engine._load_bt_data()
    holds = _holds_for(pan, SPEC["n"])     # n=20 (the product book)
    # show the last 3 rebalances' books with each name's own forward return
    for i in range(len(holds) - 3, len(holds)):
        df = pan.panels[i].set_index("company_key")
        date = pan.bdates[i].date()
        rows = []
        for ck in holds[i]:
            meta = data.get(ck, {}).get("meta", {})
            tk = meta.get("ticker", ck)
            fr = float(df.loc[ck, "fwd_ret"]) if ck in df.index else float("nan")
            rows.append((tk, fr))
        names = ", ".join(f"{tk} ({fr*100:+.0f}%)" for tk, fr in rows[:8])
        print(f"  {date}  [{len(holds[i])} names]  {names} …")
    # payload-size sanity: full history of tickers
    total_picks = sum(len(h) for h in holds)
    print(f"\n  {len(holds)} rebalances x ~{SPEC['n']} names = {total_picks} ticker strings "
          f"(~{total_picks*6/1024:.0f} KB) — trivially serializable.")


if __name__ == "__main__":
    pan = sweep()
    feasibility_probe(pan)
