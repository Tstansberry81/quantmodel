"""Daily mark-to-market of the live book against the S&P.

WHAT THIS IS FOR
The forward record shows a book and its status, but nothing between one
rebalance and the next: a position opened on the 1st sat at "—" for a month.
This marks it every trading day so the open book has a running return against
the benchmark from day one.

WHY IT NEEDS ITS OWN PRICE FEED
The deployed artifact is a static bundle — its prices stop on the day the bundle
was built (2026-07-24 for the current one) and do not move until someone
republishes. A "daily" mark computed from it would freeze silently and keep
displaying a stale number as though it were today's. So this fetches live closes
from yfinance, the same source edge_data already uses for benchmarks.

BENCHMARK: config.BENCH_SP500, which is ^SP500TR (TOTAL RETURN), deliberately
the same symbol the backtest compares against. ^GSPC is price-only and would
understate the S&P by its dividend yield — roughly 1.3%/yr of invented
outperformance, quietly, in the model's favour.

WHAT IT DOES NOT DO
No costs, no volatility target, no regime gate. This is the raw equal-weight
basket versus the raw index, which is what "how is the book doing" means. The
backtest's overlays act on a daily curve the live book has no equivalent of
yet; applying half of them here would produce a number that matches neither.
Stated on the page rather than left for a reader to assume.
"""
from __future__ import annotations
import datetime as _dt
import json
import logging
import os
import pathlib
import threading

import pandas as _pd     # already resident: every caller of this module imports it

import config

log = logging.getLogger(__name__)

def _default_state_path() -> str:
    """Where the live mark lives when EDGE_LIVE_PATH is not set.

    Beside whatever is ALREADY persisting -- never beside the code. The old
    default was the repo's own data/cache, which Render wipes on every deploy;
    that silently restarted the mark at day 1 three times on 2026-07-31 alone.
    A tracker that resets on each deploy is indistinguishable on the page from
    one that simply never accrues, which is why it survived a full day of
    verification: every reading of it was correct for a run one deploy old.

    The disk was mounted and working the whole time. EDGE_TRACKER_PATH and
    EDGE_PANEL_CACHE_DIR both pointed at it; EDGE_LIVE_PATH was added to
    render.yaml but never applied to the running service, so this one file --
    the only one holding unrecoverable data -- landed on ephemeral storage.

    So: derive from a sibling that already works rather than requiring a new
    env var for every new file. Adding a file should not be able to leave the
    service half-persistent.
    """
    for var, is_file in (("EDGE_LIVE_PATH", True),      # explicit wins
                         ("EDGE_TRACKER_PATH", True),   # forward record — same lifetime
                         ("EDGE_PANEL_CACHE_DIR", False)):
        v = os.environ.get(var)
        if not v:
            continue
        if var == "EDGE_LIVE_PATH":
            return v
        d = os.path.dirname(v) if is_file else v
        if d:
            return os.path.join(d, "edge_live.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "cache", "edge_live.json")


# The entry prices ARE the record: lose them and the since-inception return is
# unrecoverable, because yfinance will happily serve today's price but never
# what we paid.
STATE_PATH = pathlib.Path(_default_state_path())

_LOCK = threading.Lock()
# yfinance is a network call in a request path. One fetch per this many seconds;
# everything else reads the cached marks. Prices only change once a day at the
# close, so this is generous.
_MIN_REFRESH_S = int(os.environ.get("EDGE_LIVE_REFRESH_S", "900"))


def _load() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        log.warning("live tracker state unreadable; starting fresh", exc_info=True)
        return {}


def _save(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except Exception:
        log.warning("could not write live tracker state", exc_info=True)


def _closes(tickers: list[str], days: int = 10):
    """Latest daily closes for `tickers`. Returns a DataFrame or None."""
    import yfinance as yf
    df = yf.download(tickers, period=f"{days}d", interval="1d",
                     progress=False, auto_adjust=True)
    close = df["Close"] if "Close" in df else None
    if close is None or close.empty:
        return None
    return close.dropna(how="all")


def state(book: list[dict] | None = None, spec_version: str = "",
          committed_on: str | None = None) -> dict:
    """Current live mark. Starts a new run if the book changed.

    `book` is the tracker's current_book (ticker + name). Passing None reads the
    stored state without touching the network — used by callers that only want
    to display, never to start a run.

    `committed_on` (YYYY-MM-DD) is the date this exact book was first written
    down, from the forward record. When a run starts, inception belongs on that
    date rather than on whichever day this process happened to notice: the book
    was fixed then, and the closes for a past date are a matter of record, not a
    choice made with hindsight. It also makes a lost state file recoverable
    instead of silently re-dating inception to today — which reads on the page
    as a tracker that never accrues.
    """
    with _LOCK:
        st = _load()
        tickers = [str(b["ticker"]).upper() for b in (book or []) if b.get("ticker")]

        # A CHANGED BOOK STARTS A NEW RUN. The v4 book and the v5 book are
        # different portfolios; splicing their returns into one series would
        # produce a track record no strategy ever ran. Inception resets and the
        # previous run is kept under `history` rather than overwritten.
        if tickers and sorted(tickers) != sorted(st.get("book") or []):
            prev = {k: st[k] for k in ("inception", "book", "spec_version", "marks")
                    if k in st}
            close = _closes(tickers + [config.BENCH_SP500])
            if close is None:
                return st
            # Entry bar: the last close AT OR BEFORE the commit date. Falls back
            # to the latest bar when the commit date is unknown or predates the
            # fetch window -- never forward of it, which would enter at a price
            # the book could not have been bought at.
            last = close.index[-1]
            if committed_on:
                prior = close.index[close.index <= _pd.Timestamp(committed_on)]
                if len(prior):
                    last = prior[-1]
                else:
                    log.warning("live tracker: commit date %s predates the price "
                                "window; entering at %s", committed_on, last.date())
            entry = {t: float(close[t].loc[last]) for t in tickers
                     if t in close and close[t].loc[last] == close[t].loc[last]}
            if len(entry) < len(tickers):
                log.warning("live tracker: no quote for %s",
                            sorted(set(tickers) - set(entry)))
            st = {
                "inception": str(last.date()),
                "book": tickers,
                "entry": entry,
                "entry_spx": float(close[config.BENCH_SP500].loc[last]),
                "spec_version": spec_version,
                "benchmark": config.BENCH_SP500,
                "marks": [],
                "history": ([prev] + (st.get("history") or []))[:10] if prev else
                           (st.get("history") or []),
            }
            _save(st)

        if not st.get("entry"):
            return st

        # rate-limit the network call
        now = _dt.datetime.now(_dt.timezone.utc)
        fetched = st.get("fetched_at")
        if fetched:
            age = (now - _dt.datetime.fromisoformat(fetched)).total_seconds()
            if age < _MIN_REFRESH_S and st.get("marks"):
                return st

        close = _closes(list(st["entry"]) + [st.get("benchmark", config.BENCH_SP500)])
        if close is None:
            return st
        last = close.index[-1]

        rets = []
        for t, p0 in st["entry"].items():
            if t in close:
                p1 = float(close[t].loc[last])
                if p1 == p1 and p0:
                    rets.append(p1 / p0 - 1.0)
        if not rets:
            return st
        edge = sum(rets) / len(rets)              # equal weight, no rebalancing
        spx = float(close[st.get("benchmark", config.BENCH_SP500)].loc[last])
        sp = spx / st["entry_spx"] - 1.0

        # Trading sessions from inception through this bar, counted off the price
        # index rather than off how many marks we happen to have stored. Those
        # are different quantities: len(marks) measures OUR uptime, not how long
        # the position has been held. A process that missed a day, or restarted,
        # would under-report the holding period and call day 3 "day 1".
        inc = st.get("inception")
        sessions = int((close.index >= _pd.Timestamp(inc)).sum()) if inc else len(rets)

        mark = {"date": str(last.date()), "edge": edge, "sp": sp,
                "excess": edge - sp, "n_priced": len(rets), "sessions": sessions}
        # One mark per DATE, replaced in place: called twice on the same day the
        # second must update the day's number, not append a duplicate row.
        marks = [m for m in st.get("marks", []) if m["date"] != mark["date"]]
        marks.append(mark)
        st["marks"] = sorted(marks, key=lambda m: m["date"])
        st["fetched_at"] = now.isoformat()
        _save(st)
        return st


def summary(st: dict) -> dict:
    """Flat payload for the API/page. Empty dict if nothing is tracked yet."""
    marks = st.get("marks") or []
    if not marks:
        return {}
    cur = marks[-1]
    return {
        "inception": st.get("inception"),
        "as_of": cur["date"],
        # Sessions held, not marks taken. Falls back to the mark count only for
        # state written before `sessions` existed.
        "days": cur.get("sessions") or len(marks),
        "edge_ret": cur["edge"],
        "sp_ret": cur["sp"],
        "excess": cur["excess"],
        "n_priced": cur.get("n_priced"),
        "n_book": len(st.get("book") or []),
        "benchmark": st.get("benchmark"),
        "spec_version": st.get("spec_version"),
        "series": [{"d": m["date"], "e": round(m["edge"], 6), "s": round(m["sp"], 6)}
                   for m in marks],
        # No costs/overlays — see the module docstring. Said out loud so the
        # number is never mistaken for the backtest's net-of-everything curve.
        "basis": "equal-weight buy-and-hold, gross of costs, no regime or "
                 "volatility overlay; benchmark is total-return",
    }
