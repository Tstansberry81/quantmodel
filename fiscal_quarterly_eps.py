"""Pull QUARTERLY diluted EPS + earnings-announcement dates from fiscal.ai live.

For the earnings-dynamics research (round 5): builds a per-name quarterly EPS
series keyed by the earnings-announcement date, so we can compute time-series
SUE (seasonal-random-walk standardized unexpected earnings) and test PEAD without
needing analyst-estimate history. Only reported diluted EPS + report/earnings
dates are needed.

Endpoint: /v1/company/financials/income-statement/standardized?periodType=quarterly
Cached to data/cache/fiscal_q__*.json (free re-run). Writes data/cache/quarterly_eps.pkl
  { company_key: DataFrame(index=announce_date, columns=[eps, fiscalYear, fiscalQuarter]) }

Usage: .venv/Scripts/python.exe fiscal_quarterly_eps.py [--limit N]
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, requests
import config
import edge_data as D

BASE = "https://api.fiscal.ai"
EPS_ID = "income_statement_diluted_eps"
MIN_INTERVAL = config.FISCAL_MIN_INTERVAL
_last = [0.0]


def _key():
    # importing config loads .env, so the key is already in the environment --
    # and unlike reading .env directly, this doesn't crash on a host that has
    # no .env file (Render sets real environment variables).
    return config.FISCAL_API_KEY


API_KEY = _key()


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def fetch_q(ck, retries=4):
    cp = config.CACHE_DIR / f"fiscal_q__{ck.replace('/', '_')}.json"
    if cp.exists():
        try:
            j = json.loads(cp.read_text(encoding="utf-8"))
            return None if isinstance(j, dict) and "_error" in j else j
        except Exception:
            pass
    url = f"{BASE}/v1/company/financials/income-statement/standardized"
    params = {"companyKey": ck, "periodType": "quarterly", "apiKey": API_KEY}
    for a in range(retries):
        _throttle()
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (a + 1)); continue
        if r.status_code == 200:
            j = r.json(); cp.write_text(json.dumps(j), encoding="utf-8"); return j
        if r.status_code == 429:
            time.sleep(5 * (a + 1)); continue
        if r.status_code in (400, 404):
            cp.write_text(json.dumps({"_error": r.status_code}), encoding="utf-8"); return None
        time.sleep(2 * (a + 1))
    return None


def build_name(ck):
    j = fetch_q(ck)
    if not j:
        return None
    rows = []
    for d in j.get("data", []):
        ann = d.get("earningsDate") or d.get("lastSourceFilingDate")
        if not ann:
            rd = d.get("reportDate")
            ann = (pd.Timestamp(rd) + pd.Timedelta(days=45)) if rd else None  # fallback lag
        if ann is None:
            continue
        eps = (d.get("metricsValues", {}).get(EPS_ID) or {}).get("value")
        if eps is None:
            continue
        rows.append({"announce": pd.Timestamp(ann).normalize(), "eps": float(eps),
                     "fy": d.get("fiscalYear"), "fq": d.get("fiscalQuarter")})
    if len(rows) < 6:                       # need >=6 quarters for a seasonal SUE
        return None
    df = pd.DataFrame(rows).sort_values("announce").set_index("announce")
    return df[~df.index.duplicated(keep="last")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cks = list(D.load_bt_data().keys())
    if args.limit:
        cks = cks[:args.limit]
    out = {}; t0 = time.time()
    for i, ck in enumerate(cks):
        df = build_name(ck)
        if df is not None:
            out[ck] = df
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cks)}  ({len(out)} with >=6q)  {el:.0f}s, ~{el/(i+1)*len(cks)/60:.0f}min total")
    if out:
        qs = [len(v) for v in out.values()]
        yrs = [(v.index.min().year, v.index.max().year) for v in out.values()]
        print(f"\nBuilt {len(out)} names. quarters/name: min {min(qs)}, median {int(np.median(qs))}, max {max(qs)}")
        print(f"earliest announce year: {min(y[0] for y in yrs)}, latest: {max(y[1] for y in yrs)}")
    pickle.dump(out, open(config.CACHE_DIR / "quarterly_eps.pkl", "wb"))
    print(f"saved -> {config.CACHE_DIR / 'quarterly_eps.pkl'}")


if __name__ == "__main__":
    main()
