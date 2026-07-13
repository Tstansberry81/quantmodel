"""The Edge -- fundamental / quality KPI screen (2026-07-13 literature test).

Tests the 5 peer-reviewed fundamental factors from the deep-research shortlist
(see memory: fundamental-kpi-literature) as a QUALITY/EXCLUSION overlay that is
orthogonal to the price-only Edge signal. Same honest two-stage discipline as
edge_signal_ic.py: score each KPI on the whole cross-section (rank-IC + deciles
+ incremental-vs-accel + stability) BEFORE it ever touches a top-N backtest.

THE 5 KPIs  (canonical -> what we can actually build from OUR data):
  1. Gross Profitability (Novy-Marx 2013)     -> gp_margin (GP/Revenue) + roic
  2. Piotroski F-Score (Piotroski 2000)        -> fscore_lite (9-signal composite)
  3. Accruals / earnings quality (Sloan 1996)  -> accruals = net_margin - fcf_margin
  4. Asset Growth / Investment (CGS 2008)      -> rev_growth (YoY sales growth)
  5. Distress / manipulation (Ohlson/CHS)      -> distress = debt/equity (+neg-earn)
  BONUS Net share issuance (Pontiff-Woodgate)  -> shldr_yield (buybacks+divs)

DATA-AVAILABILITY REALITY (this is half of what the user asked to test):
  * fiscal.ai cached tier (data/cache/v1_company_ratios, and fund_hist in the
    backtest pickle) exposes 15 PRECOMPUTED RATIOS ONLY -- NO raw line items
    (no total assets, COGS, CFO, receivables, share count). So 3 of the 5
    canonical KPIs (GP/assets, full F-Score, asset growth) can only be built as
    RATIO PROXIES here. Upside: ~22yrs annual history (2005-2025), 1111 names
    incl. some inactive -> matches the price panel, PIT-safe (report_date + 90d).
  * yfinance DOES expose full raw statements (all line items for all 5 canonical
    KPIs) -- but only ~5yrs annual, latest-RESTATED (look-ahead), and
    survivorship-biased (current listings only). Good for a definition, useless
    for a 2005+ backtest.
  * fiscal.ai LIVE: no FISCAL_API_KEY in env -> cannot re-fetch. Frozen artifacts.

Alignment is PIT-safe and identical to edge_lib: for each rebalance date d, use
the last annual report with report_date <= d - 90 days (reporting lag). Delta
signals (dROE, dmargin, dleverage) need two reports; names with <2 get NaN.

Usage:
    .venv/Scripts/python.exe edge_test_fundamentals.py            # hold=42 (Edge clock)
    .venv/Scripts/python.exe edge_test_fundamentals.py --hold 21  # product clock
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd

import edge_lib as E
import edge_data as engine
from edge_signal_ic import scorecard   # reuse the exact honest scorecard machinery

LAG = np.timedelta64(90, "D")          # reporting lag, matches edge_lib.load_edge_panel

# All 15 fiscal.ai ratio columns we might read out of fund_hist.
_COLS = ["ratio_gross_profit_margin", "ratio_net_profit_margin", "ratio_fcf_margin",
         "ratio_return_on_invested_capital", "ratio_return_on_equity",
         "ratio_debt_to_equity", "ratio_shareholder_yield", "growth_revenue_1y",
         "growth_diluted_eps_1y", "ratio_diluted_eps", "ratio_fcf_yield",
         "ratio_earnings_yield", "ratio_ev_to_ebitda", "ratio_price_to_earnings"]


def _prep_fund():
    """Per-name {fidx, cols{col->array}} from fund_hist, for PIT asof lookups."""
    data = engine.load_bt_data()
    prep = {}
    for ck, blob in data.items():
        fh = blob.get("fund_hist")
        if fh is None or fh.empty:
            continue
        fidx = np.asarray(fh.index.values, "datetime64[ns]")
        cols = {c: np.asarray(fh[c].values, float) for c in _COLS if c in fh.columns}
        prep[ck] = {"fidx": fidx, "cols": cols}
    return prep


def _asof_pos(fidx, dlag):
    """Index of the last report with report_date <= dlag, else -1."""
    return int(np.searchsorted(fidx, dlag, side="right")) - 1


def _g(cols, name, pos):
    """Value of ratio `name` at report position `pos` (NaN-safe)."""
    a = cols.get(name)
    if a is None or pos < 0 or pos >= len(a):
        return np.nan
    return float(a[pos])


def _kpis_at(F, dlag):
    """The 5 fundamental KPIs (+ bonus) for one name as of `dlag`. PIT-safe."""
    fidx = F["fidx"]; cols = F["cols"]
    p = _asof_pos(fidx, dlag)
    if p < 0:
        return None
    q = p - 1                                   # prior annual report (for deltas)

    gpm = _g(cols, "ratio_gross_profit_margin", p)
    npm = _g(cols, "ratio_net_profit_margin", p)
    fcfm = _g(cols, "ratio_fcf_margin", p)
    roic = _g(cols, "ratio_return_on_invested_capital", p)
    roe = _g(cols, "ratio_return_on_equity", p)
    d2e = _g(cols, "ratio_debt_to_equity", p)
    shy = _g(cols, "ratio_shareholder_yield", p)
    rgr = _g(cols, "growth_revenue_1y", p)

    out = {}
    # KPI 1 -- Gross Profitability proxy (POS IC expected). Canonical is GP/assets;
    # we only have GP/revenue (margin). ROIC is the capital-efficiency robustness leg.
    out["gp_margin"] = gpm
    out["roic"] = roic
    # KPI 3 -- Accruals / earnings quality (NEG IC expected): earnings not backed
    # by cash. net_margin - fcf_margin = accrual-to-sales gap (Sloan spirit).
    out["accruals"] = (npm - fcfm) if np.isfinite(npm) and np.isfinite(fcfm) else np.nan
    # KPI 4 -- Asset Growth / Investment proxy (NEG IC expected): high growers
    # under-perform. Canonical is d(total assets); we have YoY revenue growth --
    # which the panel ALREADY carries as `rev_growth` (same asof/lag), so we don't
    # recompute it here (would collide on join); the scorecard reads it directly.
    # KPI 5 -- Distress proxy (NEG IC expected): high leverage / weak returns.
    out["distress"] = d2e
    # BONUS -- Net share issuance factor (POS IC expected): shareholder yield =
    # buybacks + dividends; high = shrinking share count = good.
    out["shldr_yield"] = shy

    # KPI 2 -- Piotroski F-Score (POS IC expected). 9 binary signals from what we
    # have; requires a prior report for the 3 delta signals, else NaN.
    if q >= 0:
        gpm0 = _g(cols, "ratio_gross_profit_margin", q)
        roe0 = _g(cols, "ratio_return_on_equity", q)
        d2e0 = _g(cols, "ratio_debt_to_equity", q)
        sig = [
            npm > 0,                                    # profitable (net)
            roe > 0,                                    # ROE positive (ROA proxy)
            roic > 0,                                   # ROIC positive
            fcfm > 0,                                   # positive cash flow (CFO>0 proxy)
            (fcfm > npm),                               # accrual quality: cash > earnings
            (roe > roe0),                               # improving profitability (dROA)
            (gpm >= gpm0),                              # improving gross margin
            (d2e <= d2e0),                              # leverage not rising
            shy > 0,                                    # not diluting shareholders
        ]
        vals = [s for s in sig if isinstance(s, (bool, np.bool_)) and
                not (isinstance(s, float) and np.isnan(s))]
        # only score if every input was present (all 9 evaluable + finite inputs)
        inputs = [npm, roe, roic, fcfm, shy, gpm, gpm0, roe0, d2e, d2e0]
        out["fscore_lite"] = float(sum(bool(s) for s in sig)) if all(
            np.isfinite(x) for x in inputs) else np.nan
    else:
        out["fscore_lite"] = np.nan
    return out


def attach(pan):
    """Join the fundamental KPI columns onto a copy of each panel df (PIT-safe)."""
    fp = _prep_fund()
    cover = {}
    panels = []
    for i, df in enumerate(pan.panels):
        dlag = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns") - LAG
        recs = {}
        for ck in df["company_key"]:
            F = fp.get(ck)
            if F is None:
                continue
            k = _kpis_at(F, dlag)
            if k is not None:
                recs[ck] = k
        add = pd.DataFrame.from_dict(recs, orient="index")
        out = df.set_index("company_key").join(add).reset_index()
        panels.append(out)
        for c in add.columns:
            cover.setdefault(c, [0, 0])
            cover[c][0] += int(out[c].notna().sum()); cover[c][1] += len(out)
    return panels, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading hold={args.hold} panel ...")
    pan = E.load_edge_panel(hold=args.hold)
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}, "
          f"avg {np.mean([len(p) for p in pan.panels]):.0f} names")
    print("Attaching fundamental KPIs from fund_hist (PIT-safe, 90d lag) ...")
    panels, cover = attach(pan)

    print("\nDATA COVERAGE (share of name-rebalance rows with a finite value):")
    for c in ["gp_margin", "roic", "fscore_lite", "accruals", "rev_growth",
              "distress", "shldr_yield"]:
        if c in cover:
            n, d = cover[c]
            print(f"    {c:<14} {n/d*100:5.1f}%   ({n:,}/{d:,})")

    print("\nEXPECTED SIGNS  (does the literature say high value = good or bad?)")
    print("    gp_margin +   roic +   fscore_lite +   shldr_yield +   (POS IC = works)")
    print("    accruals  -   rev_growth -   distress -                (NEG IC = works)")

    cols = ["gp_margin", "roic", "fscore_lite", "accruals", "rev_growth",
            "distress", "shldr_yield"]
    print("\nFUNDAMENTAL KPI SCORECARD (rank-IC vs forward return; incr = after accel):")
    scorecard(panels, pan.bdates, cols)


if __name__ == "__main__":
    main()
