"""Thin, cached, throttled client for the fiscal.ai API.

Every GET is cached to data/cache as JSON keyed by a hash of (path, params).
This protects the daily API quota and makes repeat runs / the website fast.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import requests

import config

_last_call = [0.0]  # mutable holder for last-call timestamp (throttle)


def _throttle():
    elapsed = time.time() - _last_call[0]
    if elapsed < config.FISCAL_MIN_INTERVAL:
        time.sleep(config.FISCAL_MIN_INTERVAL - elapsed)
    _last_call[0] = time.time()


def _cache_path(path: str, params: dict) -> Path:
    key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "apiKey")
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe = path.strip("/").replace("/", "_")
    return config.CACHE_DIR / f"{safe}__{h}.json"


def _fresh(p: Path) -> bool:
    if not p.exists():
        return False
    age_days = (time.time() - p.stat().st_mtime) / 86400
    return age_days <= config.CACHE_TTL_DAYS


def get(path: str, params: dict | None = None, *, retries: int = 4,
        allow_cache: bool = True) -> dict | list | None:
    """GET a fiscal.ai endpoint with caching. Returns parsed JSON or None on hard failure."""
    params = dict(params or {})
    cp = _cache_path(path, params)
    if allow_cache and _fresh(cp):
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass

    params["apiKey"] = config.FISCAL_API_KEY
    url = f"{config.FISCAL_BASE}{path}"
    last_status = None
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        last_status = r.status_code
        if r.status_code == 200:
            data = r.json()
            cp.write_text(json.dumps(data), encoding="utf-8")
            return data
        if r.status_code == 429:                      # rate limited -> back off
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (404, 400):               # missing data for this name
            # Cache the miss so we don't keep retrying it across runs.
            cp.write_text(json.dumps({"_error": r.status_code, "_body": r.text[:200]}),
                          encoding="utf-8")
            return None
        time.sleep(2 * (attempt + 1))
    return None


# ---- typed helpers ---------------------------------------------------------

def companies_page(page: int = 1) -> dict | None:
    return get("/v2/companies-list", {"pageNumber": page})


def ratios(company_key: str, ratio_ids: list[str], period_type: str = "annual") -> dict | None:
    data = get("/v1/company/ratios", {
        "companyKey": company_key,
        "ratioId": ",".join(ratio_ids),
        "periodType": period_type,
    })
    if isinstance(data, dict) and "_error" in data:
        return None
    return data


def stock_prices(company_key: str) -> list | None:
    data = get("/v1/company/stock-prices", {"companyKey": company_key})
    if isinstance(data, dict) and "_error" in data:
        return None
    return data
