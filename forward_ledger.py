"""Score the forward record from the BOOK LEDGER and PRICES -- nothing else.

WHY THIS EXISTS
The forward numbers used to be accumulated: live_tracker saved entry prices
once, appended a mark per day, and started a new run only when the panel said a
book was priced. Every failure in that chain was permanent. On 2026-09-02 the
Sharadar pull ended a session short, the 2026-09-01 book never left "entry
pending", and for a month the page marked the July/August basket from 07-30 --
reporting +7.3% for a book that was up +1.3%, double-counting August, under a
row labelled 2026-09-01. Nothing could correct it, because the wrong marks WERE
the record.

THE RULE (one rule, every row)
  entry  = the close of the first session AFTER the book date (edge_lib t+1)
  exit   = the entry close of the NEXT scoreable book -- a book is held until
           the one replacing it is bought
  return = equal-weight mean of adj_close[exit] / adj_close[entry] - 1, gross of
           costs, no overlays; benchmark is config.BENCH_SP500 (total return)
           over the same two closes.
An open book is marked to the latest close the same way. A book whose entry
session has not closed yet is PENDING and carries no number at all.

Because every number is recomputed from the ledger on each call, a late bar, a
fixed bug or a corrected price heals every row the next time the page loads.
Nothing here is stored.

PRICES: yfinance adjusted closes -- the same source and adjustment the old live
mark used, independent of the Sharadar bundle (so a stale bundle can no longer
freeze the forward record). Adjusted closes are re-read in full each refresh, so
a dividend paid during a hold is credited to the book, as the total-return
benchmark credits it to the S&P.
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from zoneinfo import ZoneInfo

import pandas as pd

import config

log = logging.getLogger(__name__)

BASIS = ("equal-weight buy-and-hold from each book's entry close, gross of "
         "costs, no regime or volatility overlay; benchmark is total-return")

# Statuses the ledger scores. STRANDED / SUPERSEDED / BACKFILLED rows are kept
# for provenance by edge_tracker_lib and are never scored -- see _forward_log.
SCOREABLE = ("OPEN", "CLOSED")

_NY = ZoneInfo("America/New_York")
# A session's close is final a little after 16:00 ET. Before that, today's bar
# from yfinance is an intraday print: fine to MARK an open book against, never
# fine to ENTER or EXIT one at.
_CLOSE_FINAL = _dt.time(16, 15)

_REFRESH_S = 15 * 60
_cache: dict = {}
_lock = threading.Lock()


def _yf_symbol(ticker: str) -> str:
    return str(ticker).upper().replace(".", "-")        # BRK.B -> BRK-B


def _download(symbols: tuple, start: str) -> pd.DataFrame | None:
    import yfinance as yf
    df = yf.download(list(symbols), start=start, interval="1d",
                     progress=False, auto_adjust=True)
    close = df["Close"] if df is not None and "Close" in df else None
    if close is None or close.empty:
        return None
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close.dropna(how="all")


def prices(symbols, start: str) -> pd.DataFrame | None:
    """Adjusted closes, cached for _REFRESH_S. A failed refresh serves the last
    good frame rather than blanking the page; a page with no frame at all says
    so instead of inventing numbers."""
    key = (tuple(sorted(set(symbols))), start)
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _REFRESH_S:
            return hit[1]
    try:
        fresh = _download(key[0], start)
    except Exception:                                      # noqa: BLE001
        log.warning("forward ledger: price refresh failed", exc_info=True)
        fresh = None
    with _lock:
        if fresh is not None:
            _cache.clear()                                 # one live key at a time
            _cache[key] = (now, fresh)
            return fresh
        return hit[1] if hit else None


def _closed_through(now: _dt.datetime | None = None) -> pd.Timestamp:
    """The last date whose close is final, in New York."""
    ny = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(_NY)
    d = pd.Timestamp(ny.date())
    return d if ny.time() >= _CLOSE_FINAL else d - pd.Timedelta(days=1)


def _ret(close: pd.DataFrame, sym: str, a: pd.Timestamp, b: pd.Timestamp):
    if sym not in close:
        return None
    p0, p1 = close[sym].get(a), close[sym].get(b)
    if p0 is None or p1 is None or p0 != p0 or p1 != p1 or not p0:
        return None
    return float(p1 / p0 - 1.0)


def score(rows: list[dict], close: pd.DataFrame | None,
          bench: str = config.BENCH_SP500,
          now: _dt.datetime | None = None) -> dict:
    """Score forward rows in place-free fashion.

    `rows` are edge_tracker_lib forward-log entries (book_date, tickers, status).
    Returns {"rows": [...copies with returns...], "held": <row or None>,
    "live": <payload for the page's live cards>}.
    """
    out = [dict(r) for r in rows]
    if close is None or bench not in close:
        return {"rows": out, "held": None, "live": {}}

    sessions = close[bench].dropna().index
    final = _closed_through(now)
    last_mark = sessions[-1]                 # may be intraday; marks only

    def entry_of(bd):
        after = sessions[sessions > pd.Timestamp(bd)]
        return after[0] if len(after) else None

    def closed(d):
        return d is not None and d <= final

    scoreable = [r for r in out if r.get("status") in SCOREABLE and r.get("book_date")]
    scoreable.sort(key=lambda r: r["book_date"])
    held = None
    for i, r in enumerate(scoreable):
        entry = entry_of(r["book_date"])
        r["entry_date"] = str(entry.date()) if entry is not None else None
        r["edge_ret"] = r["sp_ret"] = r["excess"] = None
        r["holdings"] = None
        r["mark_to_market"] = False
        if not closed(entry):
            # Names are final, the price they are bought at is not.
            r["status"] = "PENDING"
            r["entry_px_pending"] = True
            continue
        r["entry_px_pending"] = False
        nxt = scoreable[i + 1] if i + 1 < len(scoreable) else None
        exit_ = entry_of(nxt["book_date"]) if nxt else None
        if closed(exit_):
            end, r["status"] = exit_, "CLOSED"
            r["closes"] = str(exit_.date())
        else:
            # Still held: either nothing has replaced it, or its replacement
            # has not been bought yet. Marked to the latest print.
            end, r["status"] = last_mark, "OPEN"
            r["mark_to_market"] = True
            # Closes at its replacement's entry close if one is scheduled;
            # otherwise the next rebalance (already on the row) stands.
            if exit_ is not None:
                r["closes"] = str(exit_.date())
            r["as_of"] = str(last_mark.date())
            held = r
        per = [(t, _ret(close, _yf_symbol(t), entry, end)) for t in r.get("tickers") or []]
        got = [v for _, v in per if v is not None]
        sp = _ret(close, bench, entry, end)
        r["n_priced"] = len(got)
        r["holdings"] = [{"ticker": t, "ret": v} for t, v in per]
        if got and sp is not None:
            r["edge_ret"] = sum(got) / len(got)
            r["sp_ret"] = sp
            r["excess"] = r["edge_ret"] - sp

    live = {}
    if held is not None and held.get("edge_ret") is not None:
        entry = pd.Timestamp(held["entry_date"])
        syms = [_yf_symbol(t) for t in held.get("tickers") or []]
        syms = [s for s in syms if s in close]
        path = close.loc[close.index >= entry, syms + [bench]]
        base = path.iloc[0]
        e = (path[syms] / base[syms] - 1.0).mean(axis=1, skipna=True)
        s = path[bench] / base[bench] - 1.0
        live = {
            "book_date": held["book_date"],
            "inception": held["entry_date"],
            "as_of": held["as_of"],
            "days": int(len(path) - 1),          # sessions held after the entry close
            "edge_ret": held["edge_ret"],
            "sp_ret": held["sp_ret"],
            "excess": held["excess"],
            "n_priced": held["n_priced"],
            "n_book": len(held.get("tickers") or []),
            "benchmark": bench,
            "spec_version": "v5",
            "series": [{"d": str(d.date()), "e": round(float(e[d]), 6),
                        "s": round(float(s[d]), 6)} for d in path.index[1:]],
            "basis": BASIS,
        }
    return {"rows": out, "held": held, "live": live}


def score_live(rows: list[dict], bench: str = config.BENCH_SP500) -> dict:
    """score() against freshly cached prices for every ticker the ledger names."""
    books = [r for r in rows if r.get("status") in SCOREABLE and r.get("book_date")]
    if not books:
        return score(rows, None, bench)
    start = (pd.Timestamp(min(r["book_date"] for r in books))
             - pd.Timedelta(days=10)).date()
    syms = {_yf_symbol(t) for r in books for t in (r.get("tickers") or [])}
    return score(rows, prices(sorted(syms) + [bench], str(start)), bench)
