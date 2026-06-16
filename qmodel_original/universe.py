"""Build the investable universe: top-N US-listed common stocks by market cap.

companies-list is returned market-cap-sorted, so we just walk pages, filter to
US NYSE/NASDAQ active names that have the datasets we need, and take the first N.
"""
from __future__ import annotations
import json

import config
from qmodel import fiscal


def _eligible(c: dict, allow_inactive: bool) -> bool:
    if c.get("countryCode") not in config.ALLOWED_COUNTRY:
        return False
    if c.get("exchangeSymbol") not in config.ALLOWED_EXCHANGES:
        return False
    if c.get("isPreIpo"):
        return False
    ts = c.get("tradingStatus")
    if ts != "Active" and not (allow_inactive and ts == "Inactive"):
        return False
    return config.REQUIRED_DATASETS.issubset(set(c.get("availableDatasets", [])))


def _member(c: dict, rank: int | None) -> dict:
    return {
        "company_key": f"{c['exchangeSymbol']}_{c['ticker']}",
        "ticker": c["ticker"],
        "name": c["name"],
        "exchange": c["exchangeSymbol"],
        "sector": c.get("sector") or "Unknown",
        "industry": c.get("industry") or "Unknown",
        "trading_status": c.get("tradingStatus", "Active"),
        "market_cap_rank": rank,
    }


def build_universe(size: int | None = None,
                   include_inactive: bool | None = None) -> list[dict]:
    """Top-N active names by current market cap, plus (optionally) all eligible
    inactive/delisted names appended for the backtest pool.
    """
    size = size or config.UNIVERSE_SIZE
    if include_inactive is None:
        include_inactive = config.INCLUDE_INACTIVE

    active: list[dict] = []
    inactive: list[dict] = []
    page, total_pages = 1, 99
    while page <= total_pages:
        data = fiscal.companies_page(page)
        if not data:
            break
        total_pages = data.get("pagination", {}).get("totalPages", page)
        for c in data.get("data", []):
            ts = c.get("tradingStatus")
            if ts == "Active":
                if len(active) < size and _eligible(c, False):
                    active.append(_member(c, len(active) + 1))
            elif include_inactive and ts == "Inactive" and _eligible(c, True):
                inactive.append(_member(c, None))
        # keep paging if we still need inactive names or haven't filled active
        if len(active) >= size and not include_inactive:
            break
        page += 1
    return active + inactive


def save(universe: list[dict]):
    (config.ARTIFACT_DIR / "universe.json").write_text(json.dumps(universe, indent=2),
                                                       encoding="utf-8")


def load() -> list[dict]:
    p = config.ARTIFACT_DIR / "universe.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
