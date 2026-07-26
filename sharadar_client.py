"""
sharadar_client.py
==================

A small, dependency-light client for pulling Sharadar data from
Nasdaq Data Link (https://data.nasdaq.com).

Covers the tables Porter uses most:
    - SF1     : Core US Fundamentals (income / balance / cash-flow + derived)
    - SEP     : US Equity Prices (adjusted close, dividends, splits)
    - SFP     : US Fund Prices (ETFs, closed-end funds)
    - TICKERS : Ticker metadata (sector, industry, listing/delisting)
    - ACTIONS : Corporate actions (splits, dividends, spinoffs, name changes)
    - DAILY   : Daily point-in-time valuation ratios (P/E, EV/EBITDA, etc.)

Point-in-time discipline:
    - SF1 defaults to dimension='ART' (As-Reported, Trailing-Twelve-Month)
      which is the survivorship-bias-free, no-look-ahead dataset.
    - All fundamental queries filter by `datekey` (the filing date), not
      `calendardate`, so backtests never peek at data that wasn't public yet.
    - SEP prices are total-return-adjusted (closeadj) for splits & dividends.

Usage
-----
    from sharadar_client import SharadarClient
    sc = SharadarClient()  # reads NASDAQ_DATA_LINK_API_KEY from env

    # As-reported TTM fundamentals for Novartis, since 2010, filed dates only
    fund = sc.sf1("NVS", dimension="ART", datekey_gte="2010-01-01")

    # Total-return-adjusted daily prices
    px = sc.sep("NVS", date_gte="2010-01-01")

    # ETF prices (SPY, FXF, etc.)
    spy = sc.sfp("SPY", date_gte="2010-01-01")
"""

from __future__ import annotations

import io
import os
import time
from typing import Iterable, Optional, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://data.nasdaq.com/api/v3/datatables"


class SharadarError(RuntimeError):
    """Raised for any Sharadar API failure."""


class SharadarClient:
    """
    Thin wrapper around Nasdaq Data Link's Sharadar datatables endpoint.

    Parameters
    ----------
    api_key : str, optional
        Nasdaq Data Link API key. If omitted, reads NASDAQ_DATA_LINK_API_KEY
        (or QUANDL_API_KEY) from the environment.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Automatic retries on 429/5xx with exponential backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("NASDAQ_DATA_LINK_API_KEY")
            or os.environ.get("QUANDL_API_KEY")
        )
        if not self.api_key:
            raise SharadarError(
                "No API key. Set NASDAQ_DATA_LINK_API_KEY env var or pass "
                "api_key=... to SharadarClient()."
            )
        self.timeout = timeout

        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    # ------------------------------------------------------------------
    # Low-level fetch — handles pagination via cursor_id
    # ------------------------------------------------------------------
    def _get_table(
        self,
        table: str,
        params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch a full Sharadar datatable, following pagination cursors.

        Returns a pandas DataFrame with parsed date columns.
        """
        params = {k: v for k, v in (params or {}).items() if v is not None}
        params["api_key"] = self.api_key
        params["qopts.export"] = "false"  # we page in-process, not via export

        url = f"{BASE_URL}/SHARADAR/{table}.csv"

        frames = []
        cursor = None
        seen_cursors = set()
        while True:
            call_params = dict(params)
            if cursor:
                call_params["qopts.cursor_id"] = cursor

            resp = self._session.get(url, params=call_params, timeout=self.timeout)
            if resp.status_code != 200:
                raise SharadarError(
                    f"Sharadar {table} error {resp.status_code}: {resp.text[:400]}"
                )

            # Pagination cursor. The API returns this as `cursor_id` (underscore).
            # requests' header dict is case-insensitive but NOT separator-
            # insensitive, so looking up only "Cursor-ID" silently returns None
            # and every response is truncated to the 10,000-row first page --
            # a silent, data-corrupting failure. Accept every spelling.
            cursor = next(
                (resp.headers[h] for h in ("cursor_id", "Cursor-ID", "cursor-id",
                                           "Cursor_Id", "X-Cursor-Id")
                 if resp.headers.get(h)),
                None,
            )

            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if not df.empty:
                frames.append(df)

            if not cursor:
                break
            if cursor in seen_cursors:      # server repeated a cursor: stop, don't spin
                break
            seen_cursors.add(cursor)
            # Be polite to the API
            time.sleep(0.05)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        # Parse known date columns
        for col in ("date", "datekey", "calendardate", "reportperiod",
                    "lastupdated", "filingdate", "firstadded",
                    "firstpricedate", "lastpricedate", "firstquarter",
                    "lastquarter"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _join_tickers(tickers: Union[str, Iterable[str]]) -> str:
        if isinstance(tickers, str):
            return tickers
        return ",".join(sorted(set(tickers)))

    # ------------------------------------------------------------------
    # SF1 — Core US Fundamentals
    # ------------------------------------------------------------------
    def sf1(
        self,
        tickers: Union[str, Iterable[str]],
        dimension: str = "ART",
        datekey_gte: Optional[str] = None,
        datekey_lte: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Sharadar Core US Fundamentals.

        Dimensions
        ----------
        ARQ  As-Reported, Quarterly
        ARY  As-Reported, Annual
        ART  As-Reported, Trailing Twelve Months   <-- default (POINT-IN-TIME)
        MRQ  Most-Recent-Reported, Quarterly       (restated — DO NOT backtest)
        MRY  Most-Recent-Reported, Annual          (restated — DO NOT backtest)
        MRT  Most-Recent-Reported, TTM             (restated — DO NOT backtest)

        For any backtest, use ART/ARQ/ARY and filter on `datekey`
        (the SEC filing date). Never filter on `calendardate` — that's the
        period end, not when the numbers were public.
        """
        params = {
            "ticker": self._join_tickers(tickers),
            "dimension": dimension,
            "datekey.gte": datekey_gte,
            "datekey.lte": datekey_lte,
        }
        if columns:
            params["qopts.columns"] = ",".join(columns)
        df = self._get_table("SF1", params)
        if not df.empty and "datekey" in df.columns:
            df = df.sort_values(["ticker", "datekey"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # SEP — US Equity Prices (stocks)
    # ------------------------------------------------------------------
    def sep(
        self,
        tickers: Union[str, Iterable[str]],
        date_gte: Optional[str] = None,
        date_lte: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Sharadar US Equity Prices — end-of-day OHLCV + split/dividend
        adjusted close (`closeadj`) for total-return calculations.
        Includes delisted tickers (survivorship-bias-free).
        """
        params = {
            "ticker": self._join_tickers(tickers),
            "date.gte": date_gte,
            "date.lte": date_lte,
        }
        if columns:
            params["qopts.columns"] = ",".join(columns)
        df = self._get_table("SEP", params)
        if not df.empty and "date" in df.columns:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # SFP — US Fund Prices (ETFs, mutual funds, CEFs)
    # ------------------------------------------------------------------
    def sfp(
        self,
        tickers: Union[str, Iterable[str]],
        date_gte: Optional[str] = None,
        date_lte: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Sharadar Fund Prices — same schema as SEP, but for funds/ETFs."""
        params = {
            "ticker": self._join_tickers(tickers),
            "date.gte": date_gte,
            "date.lte": date_lte,
        }
        if columns:
            params["qopts.columns"] = ",".join(columns)
        df = self._get_table("SFP", params)
        if not df.empty and "date" in df.columns:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # TICKERS — metadata
    # ------------------------------------------------------------------
    def tickers(
        self,
        tickers: Optional[Union[str, Iterable[str]]] = None,
        table: Optional[str] = None,   # e.g. "SF1", "SEP", "SFP"
        sector: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> pd.DataFrame:
        """Ticker metadata — sector, industry, listing/delisting dates, etc."""
        params = {
            "ticker": self._join_tickers(tickers) if tickers else None,
            "table": table,
            "sector": sector,
            "industry": industry,
        }
        return self._get_table("TICKERS", params)

    # ------------------------------------------------------------------
    # ACTIONS — corporate actions
    # ------------------------------------------------------------------
    def actions(
        self,
        tickers: Optional[Union[str, Iterable[str]]] = None,
        date_gte: Optional[str] = None,
        date_lte: Optional[str] = None,
        action: Optional[str] = None,   # "dividend", "split", "spinoff", ...
    ) -> pd.DataFrame:
        """Splits, dividends, spinoffs, ticker changes, delistings."""
        params = {
            "ticker": self._join_tickers(tickers) if tickers else None,
            "date.gte": date_gte,
            "date.lte": date_lte,
            "action": action,
        }
        return self._get_table("ACTIONS", params)

    # ------------------------------------------------------------------
    # DAILY — point-in-time valuation ratios
    # ------------------------------------------------------------------
    def daily(
        self,
        tickers: Union[str, Iterable[str]],
        date_gte: Optional[str] = None,
        date_lte: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Daily-updated valuation ratios (P/E, P/B, EV/EBITDA, EV/Sales, etc.)
        computed at each trading day using the then-known fundamentals.
        Ideal for point-in-time valuation screens.
        """
        params = {
            "ticker": self._join_tickers(tickers),
            "date.gte": date_gte,
            "date.lte": date_lte,
        }
        df = self._get_table("DAILY", params)
        if not df.empty and "date" in df.columns:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Convenience: point-in-time joined panel
    # ------------------------------------------------------------------
    def point_in_time_panel(
        self,
        tickers: Union[str, Iterable[str]],
        start: str,
        end: Optional[str] = None,
        filing_lag_days: int = 1,
    ) -> pd.DataFrame:
        """
        Build a daily panel of prices with the most recently *filed*
        SF1/ART fundamentals as-of each trading day.

        `filing_lag_days` shifts the datekey forward N trading days to
        eliminate same-day filing look-ahead. Default of 1 matches the
        Swiss-quality backtest convention.
        """
        px = self.sep(tickers, date_gte=start, date_lte=end)
        fu = self.sf1(tickers, dimension="ART", datekey_gte=start,
                      datekey_lte=end)
        if px.empty or fu.empty:
            return px

        fu = fu.copy()
        fu["effective_date"] = (
            fu["datekey"] + pd.tseries.offsets.BDay(filing_lag_days)
        )
        fu = fu.sort_values(["ticker", "effective_date"])

        out = pd.merge_asof(
            px.sort_values("date"),
            fu.drop(columns=["datekey"]),
            left_on="date",
            right_on="effective_date",
            by="ticker",
            direction="backward",
        )
        return out


# ---------------------------------------------------------------------------
# Quick smoke test — `python sharadar_client.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    try:
        sc = SharadarClient()
    except SharadarError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)

    print("Fetching NVS SF1 ART (last 4 filings)...")
    nvs = sc.sf1("NVS", dimension="ART", datekey_gte="2024-01-01")
    print(nvs[["ticker", "datekey", "revenue", "netinc", "roic"]].tail())

    print("\nFetching NVS SEP prices (last 5 days)...")
    px = sc.sep("NVS", date_gte="2025-01-01")
    print(px[["ticker", "date", "close", "closeadj"]].tail())

    print("\nOK — Sharadar client is working.")
