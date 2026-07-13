"""The Edge -- CANONICAL fundamental KPI test (raw line items from fiscal.ai live).

The first pass (edge_test_fundamentals.py) could only build RATIO PROXIES from
the frozen 15-ratio tier, which undersold 3 of the 5 factors. This uses the full
standardized statements pulled by fiscal_fundamentals.py -> fund_canonical.pkl to
build the CANONICAL peer-reviewed KPIs and re-run the same honest scorecard.

THE 5 (now canonical) + bonus:
  1. Gross Profitability (Novy-Marx 2013)  = gross_profit / total_assets        [POS]
  2. Piotroski F-Score (2000)              = 9 binary signals (ROA/CFO/accrual/
                                             leverage/liquidity/shares/margin/turn) [POS]
  3. Accruals / earnings quality (Sloan/   = (net_income - cfo) / avg_assets      [NEG]
     Hribar-Collins cash-flow accruals)
  4. Asset Growth (Cooper-Gulen-Schill 08) = d(total_assets) YoY                  [NEG]
  5. Distress / manipulation               = Ohlson O-Score (1980) AND
                                             Beneish M-Score (1999)               [NEG]
  bonus Net share issuance (Pontiff-       = d(diluted_shares) YoY               [NEG]
        Woodgate 2008)

PIT-safe: fund_canonical is indexed by the filing date (lastSourceFilingDate), so
for rebalance date d we use the last annual report with pit <= d -- no look-ahead,
no extra lag needed. Delta/growth signals use the prior annual report.

Usage:
    .venv/Scripts/python.exe edge_test_fund_canonical.py            # hold=42
    .venv/Scripts/python.exe edge_test_fund_canonical.py --hold 21 --pkl data/cache/fund_canonical.pkl
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import argparse
import pickle
import numpy as np, pandas as pd

import edge_lib as E
from edge_signal_ic import scorecard

_L = np.log

KPI_COLS = ["gp_assets", "fscore", "accruals", "asset_growth",
            "ohlson_o", "beneish_m", "share_iss"]


def _safe(x):
    try:
        x = float(x); return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _kpis(cur: dict, prev: dict | None) -> dict:
    """Canonical KPIs from one name's current + prior annual line items."""
    g = lambda d, k: _safe(d.get(k)) if d else np.nan
    ta = g(cur, "total_assets"); rev = g(cur, "revenue"); gp = g(cur, "gross_profit")
    cogs = g(cur, "cogs"); ni = g(cur, "net_income"); cfo = g(cur, "cfo")
    ca = g(cur, "current_assets"); cl = g(cur, "current_liabilities")
    ltd = g(cur, "long_term_debt"); tl = g(cur, "total_liabilities")
    rec = g(cur, "receivables"); ppe = g(cur, "net_ppe"); dep = g(cur, "dep_amort_cf")
    sga = g(cur, "sga"); sh = g(cur, "diluted_shares")
    if not np.isfinite(gp) and np.isfinite(rev) and np.isfinite(cogs):
        gp = rev - cogs

    out = {}
    # KPI 1 -- Gross Profitability (Novy-Marx): GP / total assets. [POS IC]
    out["gp_assets"] = gp / ta if np.isfinite(gp) and ta else np.nan
    # KPI 3 -- Accruals (cash-flow accruals, Hribar-Collins form of Sloan). [NEG IC]
    ta0 = g(prev, "total_assets") if prev else np.nan
    avg_ta = np.nanmean([ta, ta0]) if np.isfinite(ta0) else ta
    out["accruals"] = (ni - cfo) / avg_ta if np.isfinite(ni) and np.isfinite(cfo) and avg_ta else np.nan
    # KPI 4 -- Asset Growth (Cooper-Gulen-Schill): d(total assets) YoY. [NEG IC]
    out["asset_growth"] = (ta / ta0 - 1) if np.isfinite(ta0) and ta0 else np.nan
    # bonus -- Net share issuance (Pontiff-Woodgate): diluted share-count growth. [NEG IC]
    sh0 = g(prev, "diluted_shares") if prev else np.nan
    out["share_iss"] = (sh / sh0 - 1) if np.isfinite(sh0) and sh0 else np.nan

    # KPI 2 -- Piotroski F-Score (9 signals). [POS IC]  needs prior year.
    if prev:
        ni0 = g(prev, "net_income"); rev0 = g(prev, "revenue"); gp0 = g(prev, "gross_profit")
        ca0 = g(prev, "current_assets"); cl0 = g(prev, "current_liabilities")
        ltd0 = g(prev, "long_term_debt")
        cogs0 = g(prev, "cogs")
        if not np.isfinite(gp0) and np.isfinite(rev0) and np.isfinite(cogs0):
            gp0 = rev0 - cogs0
        roa = ni / ta if ta else np.nan; roa0 = ni0 / ta0 if ta0 else np.nan
        lev = ltd / ta if ta else np.nan; lev0 = ltd0 / ta0 if ta0 else np.nan
        cr = ca / cl if cl else np.nan; cr0 = ca0 / cl0 if cl0 else np.nan
        gm = gp / rev if rev else np.nan; gm0 = gp0 / rev0 if rev0 else np.nan
        at = rev / ta if ta else np.nan; at0 = rev0 / ta0 if ta0 else np.nan
        sig = [roa > 0, cfo > 0, roa > roa0, cfo > ni,        # profitability + accrual
               lev < lev0, cr > cr0, sh <= sh0,               # leverage / liquidity / dilution
               gm > gm0, at > at0]                            # efficiency
        inputs = [roa, roa0, cfo, ni, lev, lev0, cr, cr0, sh, sh0, gm, gm0, at, at0]
        out["fscore"] = float(sum(bool(s) for s in sig)) if all(np.isfinite(x) for x in inputs) else np.nan
    else:
        out["fscore"] = np.nan

    # KPI 5a -- Ohlson O-Score (1980), statement-only (GNP index dropped ~ constant). [NEG IC]
    wc = ca - cl if np.isfinite(ca) and np.isfinite(cl) else np.nan
    ffo = cfo  # funds from operations ~ operating cash flow
    ni0 = g(prev, "net_income") if prev else np.nan
    if all(np.isfinite(x) and (x != 0 or True) for x in [ta, tl, wc, cl, ca, ni, ffo]) and ta > 0 and cl and ca and tl:
        chin = ((ni - ni0) / (abs(ni) + abs(ni0))) if (prev and np.isfinite(ni0) and (abs(ni) + abs(ni0)) > 0) else 0.0
        o = (-1.32 - 0.407 * _L(ta) + 6.03 * (tl / ta) - 1.43 * (wc / ta)
             + 0.0757 * (cl / ca) - 1.72 * (1.0 if tl > ta else 0.0) - 2.37 * (ni / ta)
             - 1.83 * (ffo / tl if tl else 0.0)
             + 0.285 * (1.0 if (np.isfinite(ni0) and ni < 0 and ni0 < 0) else 0.0) - 0.521 * chin)
        out["ohlson_o"] = float(o)
    else:
        out["ohlson_o"] = np.nan

    # KPI 5b -- Beneish M-Score (1999), 8 indices, needs prior year. [NEG IC]
    if prev:
        rec0 = g(prev, "receivables"); ppe0 = g(prev, "net_ppe"); dep0 = g(prev, "dep_amort_cf")
        sga0 = g(prev, "sga"); ta0b = ta0; cl0b = cl0 if 'cl0' in dir() else g(prev, "current_liabilities")
        ltd0b = g(prev, "long_term_debt"); ca0b = g(prev, "current_assets")
        def ratio(a, b): return a / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan
        dsri = ratio(rec / rev if rev else np.nan, rec0 / rev0 if (prev and rev0) else np.nan) \
            if np.isfinite(rec) and np.isfinite(rec0) else np.nan
        gmi = ratio(gm0, gm) if np.isfinite(gm) and np.isfinite(gm0) else np.nan
        aqi_t = 1 - (ca + ppe) / ta if (np.isfinite(ca) and np.isfinite(ppe) and ta) else np.nan
        aqi_0 = 1 - (ca0b + ppe0) / ta0 if (np.isfinite(ca0b) and np.isfinite(ppe0) and ta0) else np.nan
        aqi = ratio(aqi_t, aqi_0)
        sgi = ratio(rev, rev0)
        depi = ratio(dep0 / (dep0 + ppe0) if (np.isfinite(dep0) and (dep0 + ppe0)) else np.nan,
                     dep / (dep + ppe) if (np.isfinite(dep) and (dep + ppe)) else np.nan)
        sgai = ratio(sga / rev if rev else np.nan, sga0 / rev0 if (prev and rev0) else np.nan)
        lvgi = ratio((ltd + cl) / ta if ta else np.nan,
                     (ltd0b + cl0b) / ta0 if (np.isfinite(ltd0b) and np.isfinite(cl0b) and ta0) else np.nan)
        tata = (ni - cfo) / ta if (np.isfinite(ni) and np.isfinite(cfo) and ta) else np.nan
        idx = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
        if sum(np.isfinite(x) for x in idx) >= 6:
            f = lambda x, d=1.0: x if np.isfinite(x) else d      # neutral fill for the 1-2 missing
            m = (-4.84 + 0.92 * f(dsri) + 0.528 * f(gmi) + 0.404 * f(aqi) + 0.892 * f(sgi)
                 + 0.115 * f(depi) - 0.172 * f(sgai) + 4.679 * f(tata, 0.0) - 0.327 * f(lvgi))
            out["beneish_m"] = float(m)
        else:
            out["beneish_m"] = np.nan
    else:
        out["beneish_m"] = np.nan
    return out


def attach(pan, fund: dict):
    """Join canonical KPI columns onto a copy of each panel df (PIT-safe on filing date)."""
    # pre-extract per name: sorted list of (pit_ts, row_dict)
    prep = {}
    for ck, df in fund.items():
        idx = df.index.values.astype("datetime64[ns]")
        recs = df.to_dict("records")
        prep[ck] = (idx, recs)
    cover = {}; panels = []
    for i, pdf in enumerate(pan.panels):
        d = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        out = {}
        for ck in pdf["company_key"]:
            pr = prep.get(ck)
            if pr is None:
                continue
            idx, recs = pr
            pos = int(np.searchsorted(idx, d, side="right")) - 1
            if pos < 0:
                continue
            cur = recs[pos]; prev = recs[pos - 1] if pos >= 1 else None
            out[ck] = _kpis(cur, prev)
        add = pd.DataFrame.from_dict(out, orient="index").reindex(columns=KPI_COLS)
        joined = pdf.set_index("company_key").join(add).reset_index()
        panels.append(joined)
        for c in add.columns:
            cover.setdefault(c, [0, 0])
            cover[c][0] += int(joined[c].notna().sum()); cover[c][1] += len(joined)
    return panels, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=42)
    ap.add_argument("--pkl", default="data/cache/fund_canonical.pkl")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        fund = pickle.load(f)
    print(f"Loaded canonical fundamentals: {len(fund)} names from {args.pkl}")
    print(f"Loading hold={args.hold} panel ...")
    pan = E.load_edge_panel(hold=args.hold)
    print(f"Panel: {pan.T} rebalances, avg {np.mean([len(p) for p in pan.panels]):.0f} names")
    print("Attaching CANONICAL KPIs (PIT-safe on filing date) ...")
    panels, cover = attach(pan, fund)

    cols = KPI_COLS
    print("\nDATA COVERAGE (share of name-rebalance rows with a finite value):")
    for c in cols:
        if c in cover:
            n, d = cover[c]; print(f"    {c:<14} {n/d*100:5.1f}%   ({n:,}/{d:,})")

    print("\nEXPECTED SIGNS:  gp_assets +   fscore +   (POS IC = works)")
    print("                 accruals -   asset_growth -   ohlson_o -   beneish_m -   share_iss -   (NEG IC = works)")
    print("\nCANONICAL FUNDAMENTAL KPI SCORECARD (rank-IC vs fwd ret; incr = after accel):")
    scorecard(panels, pan.bdates, cols)


if __name__ == "__main__":
    main()
