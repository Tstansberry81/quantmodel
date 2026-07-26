"""Canonical fundamentals from fiscal.ai LIVE -- raw standardized line items.

The frozen backtest pickle only carries 15 precomputed RATIOS. This module pulls
the FULL standardized financial statements (income statement / balance sheet /
cash flow) that fiscal.ai exposes on the live API, so we can build the CANONICAL
peer-reviewed KPIs (Novy-Marx GP/assets, 9-signal Piotroski, Sloan balance-sheet
accruals, Cooper-Gulen-Schill asset growth, Ohlson/Beneish distress) instead of
the ratio proxies that undersold 3 of the 5 in the first pass.

Endpoint: /v1/company/financials/{statementType}/standardized (apiKey query param)
  statementType in {income-statement, balance-sheet, cash-flow-statement}
Each response: data = list of PERIODS, each with metricsValues{id:{value,...}} and
  reportDate / earningsDate / lastSourceFilingDate / isRestated / isPointInTime.

PIT alignment: we key each annual period by its lastSourceFilingDate (when the
filing actually hit the tape) -- NOT the fiscal period end -- so a backtest that
looks up "latest filing as of date d" has NO look-ahead. Falls back to
earningsDate, then reportDate+90d, if a filing date is missing.

Everything is cached to data/cache/fiscal_fin__*.json (one file per name+stmt),
so a re-run is free. Throttled to stay under the 50 req/min free tier.

Usage:
    .venv/Scripts/python.exe fiscal_fundamentals.py --limit 50    # validation batch
    .venv/Scripts/python.exe fiscal_fundamentals.py               # full universe
    -> writes data/cache/fund_canonical.pkl  {company_key: DataFrame(pit_date x line_items)}
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
import edge_data as D

BASE = "https://api.fiscal.ai"
STMTS = ("income-statement", "balance-sheet", "cash-flow-statement")
MIN_INTERVAL = config.FISCAL_MIN_INTERVAL          # 1.25s -> < 50/min
_last_call = [0.0]

# The standardized line items we need for the 5 canonical KPIs. Kept small and
# explicit so the panel is compact and the intent is auditable.
LINE_ITEMS = {
    # income statement
    "income_statement_total_revenues": "revenue",
    "income_statement_cost_of_sales": "cogs",
    "income_statement_gross_profit": "gross_profit",
    "income_statement_selling_general_and_administrative_expenses": "sga",
    "income_statement_net_income_attributable_to_common_shareholders": "net_income",
    "income_statement_diluted_weighted_average_shares_outstanding": "diluted_shares",
    # balance sheet
    "balance_sheet_total_assets": "total_assets",
    "balance_sheet_total_current_assets": "current_assets",
    "balance_sheet_total_current_liabilities": "current_liabilities",
    "balance_sheet_total_trade_receivables": "receivables",
    "balance_sheet_inventories": "inventory",
    "balance_sheet_long_term_debt": "long_term_debt",
    "balance_sheet_total_liabilities": "total_liabilities",
    "balance_sheet_total_common_shareholders_equity": "equity",
    "balance_sheet_total_cash_and_cash_equivalents": "cash",
    "balance_sheet_net_property_plant_and_equipment": "net_ppe",
    # cash flow
    "cash_flow_statement_cash_from_operating_activities": "cfo",
    "cash_flow_statement_depreciation_and_amortization": "dep_amort_cf",
    "cash_flow_statement_net_issuance_or_repurchases_of_common_shares": "net_share_issuance",
}


def _key() -> str:
    # importing config loads .env, so the key is already in the environment --
    # and unlike reading .env directly, this doesn't crash on a host that has
    # no .env file (Render sets real environment variables).
    return config.FISCAL_API_KEY


API_KEY = _key()


def _throttle():
    dt = time.time() - _last_call[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def _cache_path(ck: str, stmt: str) -> Path:
    safe = ck.replace("/", "_")
    return config.CACHE_DIR / f"fiscal_fin__{safe}__{stmt}.json"


def fetch_statement(ck: str, stmt: str, *, retries: int = 4) -> dict | None:
    """Standardized annual statement for one company, cached to disk."""
    cp = _cache_path(ck, stmt)
    if cp.exists():
        try:
            j = json.loads(cp.read_text(encoding="utf-8"))
            return None if isinstance(j, dict) and "_error" in j else j
        except Exception:
            pass
    url = f"{BASE}/v1/company/financials/{stmt}/standardized"
    params = {"companyKey": ck, "periodType": "annual", "apiKey": API_KEY}
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1)); continue
        if r.status_code == 200:
            j = r.json(); cp.write_text(json.dumps(j), encoding="utf-8"); return j
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1)); continue
        if r.status_code in (400, 404):
            cp.write_text(json.dumps({"_error": r.status_code}), encoding="utf-8")
            return None
        time.sleep(2 * (attempt + 1))
    return None


def _pit_date(period: dict) -> pd.Timestamp | None:
    """When did this period's data become public? Filing date > earnings date >
    reportDate + 90d fallback. This is the date a backtest may 'see' it."""
    for k in ("lastSourceFilingDate", "earningsDate"):
        v = period.get(k)
        if v:
            try:
                return pd.Timestamp(v).normalize()
            except Exception:
                pass
    rd = period.get("reportDate")
    if rd:
        try:
            return (pd.Timestamp(rd) + pd.Timedelta(days=90)).normalize()
        except Exception:
            pass
    return None


def _values(period: dict) -> dict:
    mv = period.get("metricsValues") or {}
    out = {}
    for fid, short in LINE_ITEMS.items():
        e = mv.get(fid)
        out[short] = (e or {}).get("value") if isinstance(e, dict) else None
    return out


def build_name(ck: str) -> pd.DataFrame | None:
    """One company's fundamentals as a DataFrame indexed by PIT date, columns =
    the LINE_ITEMS short names. Merges the three statements on periodId."""
    per = {}  # periodId -> {pit, fiscalYear, restated, values...}
    got_any = False
    for stmt in STMTS:
        j = fetch_statement(ck, stmt)
        if not j:
            continue
        got_any = True
        for period in j.get("data", []):
            pid = period.get("periodId")
            if pid is None:
                continue
            rec = per.setdefault(pid, {"pit": _pit_date(period),
                                       "fiscalYear": period.get("fiscalYear"),
                                       "restated": period.get("isRestated")})
            rec.update({k: v for k, v in _values(period).items() if v is not None})
    if not got_any or not per:
        return None
    rows = []
    for pid, rec in per.items():
        if rec.get("pit") is None:
            continue
        row = {short: rec.get(short) for short in LINE_ITEMS.values()}
        row["pit"] = rec["pit"]; row["fiscalYear"] = rec["fiscalYear"]
        row["restated"] = rec["restated"]
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("pit").set_index("pit")
    return df[~df.index.duplicated(keep="last")]


def build(limit: int | None = None) -> dict:
    data = D.load_bt_data()
    cks = list(data.keys())
    if limit:
        cks = cks[:limit]
    out = {}
    t0 = time.time()
    for i, ck in enumerate(cks):
        df = build_name(ck)
        if df is not None and len(df) >= 2:
            out[ck] = df
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cks)} names  ({len(out)} with >=2yrs)  "
                  f"{el:.0f}s elapsed, ~{el/(i+1)*len(cks)/60:.0f}min total")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N names (validation batch)")
    ap.add_argument("--out", default=str(config.CACHE_DIR / "fund_canonical.pkl"))
    args = ap.parse_args()

    print(f"API key: {API_KEY[:8]}...{API_KEY[-4:]}  | statements: {STMTS}")
    print(f"Line items: {len(LINE_ITEMS)}  | limit: {args.limit or 'ALL'}")
    out = build(limit=args.limit)

    # coverage report on the key line items
    keyitems = ["revenue", "gross_profit", "total_assets", "cfo", "net_income",
                "receivables", "current_assets", "long_term_debt", "diluted_shares"]
    print(f"\nBuilt {len(out)} names with >=2 annual periods.")
    if out:
        yrs = [len(df) for df in out.values()]
        print(f"Annual periods/name: min {min(yrs)}, median {int(np.median(yrs))}, max {max(yrs)}")
        spans = [(df.index.min().year, df.index.max().year) for df in out.values()]
        print(f"Earliest PIT year: {min(s[0] for s in spans)}, latest: {max(s[1] for s in spans)}")
        print("\nLine-item coverage (share of name-periods with a value):")
        allrows = pd.concat(out.values())
        for it in keyitems:
            if it in allrows.columns:
                print(f"    {it:<18} {allrows[it].notna().mean()*100:5.1f}%")
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
