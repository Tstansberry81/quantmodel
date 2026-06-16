"""The Edge -- SURVIVORSHIP BIAS quantification.

Jared's caveat: the Russell-1000 PROXY universe (top-N by point-in-time market
cap from our cached data) is survivorship-inflated. It contains ~1000 active
names and only ~111 inactive ones (mostly recent M&A), and NO long-dead names
(Lehman, Enron, Washington Mutual, Wachovia, Circuit City, Bear Stearns, ...).
A real point-in-time Russell 1000 loses ~5-7%/yr of names to delisting/removal,
so the historical pool here is missing the vast majority of "dead" names.

This script:
  1. Quantifies the gap (how many dead names we have vs should have).
  2. Bounds the bias by comparing FULL POOL vs ACTIVE-ONLY versions of the
     strategy over 1Y/2Y/5Y/MAX. The gap is a measurable LOWER BOUND -- the
     truly missing dead names would drag returns lower still.
  3. Reports a recommended real-data path + honest CAGR haircut.

Run: .venv\\Scripts\\python.exe edge_test_survivorship.py
Does NOT modify edge_lib.py or qmodel/.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from qmodel import engine

PPY = E.PPY


def _status_map():
    data = engine._load_bt_data()
    out = {}
    for ck, blob in data.items():
        out[ck] = str(blob.get("meta", {}).get("trading_status", "Unknown"))
    return out


def _is_active(ck, smap):
    return smap.get(ck, "Active") == "Active"


# ---------------------------------------------------------------------------
# 1. QUANTIFY THE GAP
# ---------------------------------------------------------------------------
def quantify_gap(pan, smap):
    print("=" * 78)
    print("1. HOW SURVIVORSHIP-LIMITED IS THE POOL?")
    print("=" * 78)

    data = engine._load_bt_data()
    n_total = len(data)
    n_inactive = sum(1 for ck in data if not _is_active(ck, smap))
    n_active = n_total - n_inactive
    print(f"  Cached pool                 : {n_total} names")
    print(f"    Active (survivors today)  : {n_active}")
    print(f"    Inactive (delisted/M&A)   : {n_inactive}  "
          f"({n_inactive/n_total*100:.1f}% of pool)")

    # how many distinct names ever appear in the point-in-time top-1000
    ever_in_top = set()
    inactive_ever_in_top = set()
    for df in pan.panels:
        cks = set(df["company_key"])
        ever_in_top |= cks
        inactive_ever_in_top |= {c for c in cks if not _is_active(c, smap)}
    print(f"\n  Names that ever entered the PIT top-{E.UNIVERSE}: {len(ever_in_top)}")
    print(f"    of which inactive/delisted: {len(inactive_ever_in_top)} "
          f"({len(inactive_ever_in_top)/max(len(ever_in_top),1)*100:.1f}%)")

    # span of the backtest
    yrs = (pan.bdates[-1] - pan.bdates[0]).days / 365.25
    print(f"\n  Backtest span: {pan.bdates[0].date()} -> {pan.bdates[-1].date()} "
          f"({yrs:.1f} yrs, {pan.T} rebalances)")

    # how often the strategy's actual top-10 picked a name that is now dead
    holds = [E.select_topN(df) for df in pan.panels]
    picks_total = sum(len(h) for h in holds)
    picks_dead = sum(1 for h in holds for c in h if not _is_active(c, smap))
    periods_with_dead = sum(1 for h in holds
                            if any(not _is_active(c, smap) for c in h))
    print(f"\n  Strategy top-10 picks over all rebalances: {picks_total}")
    print(f"    picks that LATER delisted/went inactive  : {picks_dead} "
          f"({picks_dead/max(picks_total,1)*100:.1f}%)")
    print(f"    rebalances holding >=1 such name          : {periods_with_dead}/{pan.T}")

    # ----- the "missing dead names" estimate -----------------------------
    # A real Russell 1000 turns over ~5-7%/yr; a large chunk of removals are
    # delistings (bankruptcy, going-private, acquisitions, index relegation).
    # Use a conservative 4%/yr DELISTING rate (lower end -- removals also
    # include index reshuffles that are not "deaths").
    delist_rate = 0.04
    expected_deaths = E.UNIVERSE * delist_rate * yrs
    have_deaths = len(inactive_ever_in_top)
    print(f"\n  --- 'Missing dead names' estimate ---")
    print(f"  A real Russell 1000 loses ~5-7%/yr of names to delisting/removal.")
    print(f"  At a conservative {delist_rate*100:.0f}%/yr delisting rate over {yrs:.1f} yrs,")
    print(f"  a point-in-time R1000 would have seen ~{expected_deaths:.0f} distinct")
    print(f"  delisted names pass through the index.")
    print(f"  We actually have {have_deaths} dead names that touched the top-{E.UNIVERSE}.")
    captured = have_deaths / max(expected_deaths, 1)
    print(f"  => we capture roughly {captured*100:.0f}% of the dead names a real")
    print(f"     index would contain; ~{(1-captured)*100:.0f}% of 'deaths' are MISSING.")
    print(f"  (And the ones we DO have are mostly recent M&A, not bankruptcies --")
    print(f"   acquired names often exit FLAT/UP, the worst losers are absent.)")
    return holds


# ---------------------------------------------------------------------------
# 2. BOUND THE BIAS: full pool vs active-only
# ---------------------------------------------------------------------------
def active_only_panel(pan, smap):
    """Return a shallow copy of the panel whose per-rebalance frames are
    filtered to ACTIVE-ONLY names (today's survivors). Most biased version."""
    new_panels = []
    for df in pan.panels:
        mask = df["company_key"].map(lambda c: _is_active(c, smap))
        new_panels.append(df[mask].copy())
    return E.EdgePanel(new_panels, pan.bdates, pan.spxf, pan.ma200_on,
                       pan.mkt_daily, pan.sectors)


def bound_bias(pan, smap, holds_full):
    print("\n" + "=" * 78)
    print("2. BOUND THE BIAS -- FULL POOL vs ACTIVE-ONLY (measurable lower bound)")
    print("=" * 78)
    print("  (a) FULL POOL   : includes the 111 inactive names we DO have")
    print("  (b) ACTIVE-ONLY : drop trading_status != Active = 'today's survivors'")
    print("  Gap (a - b) is a LOWER BOUND on survivorship effect; truly missing")
    print("  dead names would push real returns BELOW even the full-pool line.\n")

    # full pool: reuse holds computed in part 1
    rets_full = E.period_returns(pan, holds_full)

    # active-only: re-rank within survivors-only universe each rebalance
    pan_act = active_only_panel(pan, smap)
    holds_act = [E.select_topN(df) for df in pan_act.panels]
    rets_act = E.period_returns(pan_act, holds_act)

    E.eval_windows(pan, rets_full, "(a) FULL POOL  -- 3mo momentum, EW top-10")
    print()
    E.eval_windows(pan, rets_act, "(b) ACTIVE-ONLY -- same strategy, survivors universe")

    # explicit CAGR gap per window
    print("\n  CAGR GAP (full - active-only) -- the measurable survivorship lower bound:")
    print(f"    {'window':<8}{'full':>9}{'active':>9}{'gap':>9}")
    for wname, k in (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None)):
        kk = len(rets_full) if k is None else min(k, len(rets_full))
        cf, _, _ = E.perf(rets_full[-kk:])
        ca, _, _ = E.perf(rets_act[-kk:])
        print(f"    {wname:<8}{cf*100:8.1f}%{ca*100:8.1f}%{(ca-cf)*100:>+8.1f}%")
    print("  (active-only CAGR sits ABOVE full-pool by this much -- and the real,")
    print("   fully-de-biased number sits BELOW full-pool by more than this gap.)")
    return rets_full, rets_act


# ---------------------------------------------------------------------------
# 3. RECOMMENDATION
# ---------------------------------------------------------------------------
def recommend(rets_full):
    print("\n" + "=" * 78)
    print("3. REAL-DATA PATH + HONEST HAIRCUT")
    print("=" * 78)
    cf, _, _ = E.perf(rets_full)
    print(f"  Headline full-pool MAX CAGR: {cf*100:.1f}%\n")

    print("  POINT-IN-TIME R1000 CONSTITUENTS + DELISTED RETURNS -- options:")
    print("  -------------------------------------------------------------------")
    print("  1) CRSP (via WRDS)            -- gold standard. Survivor-bias-free")
    print("       daily returns incl. delisting returns + historical index")
    print("       membership. Cost: WRDS academic ~free at a university, else")
    print("       commercial ~$15-50k/yr. Effort: medium (well-documented).")
    print("  2) Norgate Data              -- retail-priced PIT index constituents")
    print("       (incl. Russell 1000) + delisted symbols w/ adjusted prices.")
    print("       Cost: ~$50-80/mo (Platinum w/ historical constituents).")
    print("       Effort: low. BEST cost/effort for this project.")
    print("  3) Sharadar / Nasdaq Data Link (SF1+SEP+TICKERS+ACTIONS) -- includes")
    print("       delisted tickers + corporate actions; index membership is")
    print("       partial. Cost: ~$40-150/mo. Effort: low-medium.")
    print("  4) FTSE Russell historical constituents -- official membership lists")
    print("       (reconstitution-date snapshots) but NO returns; pair with a")
    print("       delisted-returns source. Cost: license, $$$. Effort: high.")
    print("  RECOMMENDED: Norgate (cheap, PIT constituents + delisted returns,")
    print("       low integration effort) -- or CRSP if WRDS access is available.\n")

    print("  SURVIVORSHIP DISCOUNT TO APPLY:")
    print("  -------------------------------------------------------------------")
    print("  Standard academic finding: survivorship bias inflates equity")
    print("  backtest returns by ~1-4%/yr (Brown/Goetzmann/Ross; Elton-Gruber-")
    print("  Blake; Shumway delisting-returns work). The effect is LARGER for:")
    print("    - momentum strategies (losers that delist are exactly the names")
    print("      a momentum filter would have ALREADY dropped -- partial offset --")
    print("      BUT crash/blowup names that round-trip UP then DOWN still hurt),")
    print("    - smaller / lower-quality names, and high-turnover books.")
    print("  This pool is WORSE than a typical survivor-bias dataset: we are")
    print("  missing ~80-90% of the dead names entirely (only recent M&A present).")
    print()
    print("  HONEST HAIRCUT (mental adjustment to headline CAGR):")
    print(f"    - Conservative : -2%/yr  ->  ~{(cf-0.02)*100:.1f}% CAGR")
    print(f"    - Central      : -3%/yr  ->  ~{(cf-0.03)*100:.1f}% CAGR")
    print(f"    - Aggressive   : -4-5%/yr->  ~{(cf-0.045)*100:.1f}% CAGR")
    print("  Use the central -3%/yr as the default 'survivorship discount' on the")
    print("  headline number until real PIT R1000 + delisting data is wired in.")


def main():
    print("Building Edge trading panel (one-time, ~1-2 min) ...\n")
    pan = E.load_edge_panel()
    smap = _status_map()
    holds_full = quantify_gap(pan, smap)
    rets_full, rets_act = bound_bias(pan, smap, holds_full)
    recommend(rets_full)


if __name__ == "__main__":
    main()
