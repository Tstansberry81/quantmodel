"""Decide whether TODAY is the day to run the monthly rebalance.

WHY A GATE INSTEAD OF A CRON EXPRESSION
---------------------------------------
The model's clock is "the first TRADING day of every month" (edge_lib._rebal_grid).
Cron cannot express a trading day: the 1st is a weekend or a holiday about a
third of the time, and the offset from the 1st to the first session moves every
month. So the workflow wakes on every weekday of the first eight days and this
script decides whether to proceed.

WHAT IT WAITS FOR
-----------------
Not the book date -- the ENTRY BAR. edge_lib prices entry at the t+1 close
(`n_fwd < 1: continue` in _build_edge_panel), so the panel cannot produce a live
book for rebalance date d until the session AFTER d has closed. Running before
that yields an empty live panel, which is worse than not running: `live_date` is
still set while `_current_book` falls back to the previous rebalance, so the
forward record gets a snapshot dated d carrying the PREVIOUS month's basket.
(edge_lib.py is patched alongside this so that mislabel cannot happen, but the
gate still waits, because a book we cannot compute is not a book.)

LATENESS
--------
edge_tracker_lib._logged_late marks any book written down more than 3 CALENDAR
days after its own date as BACKFILLED -- kept for provenance, never scored. So
this wants to fire on the evening of the entry bar (lag 1 in the common case).
The residual: when the first trading day of a month is a FRIDAY, the entry bar
is the following Monday and the earliest possible lag is 3 -- right at the
boundary. If Sharadar has not published Monday's close by the time we run, the
retry lands Tuesday at lag 4 and the row is BACKFILLED. That case is reported
loudly rather than silently accepted; see DEPLOY.md.

Exit codes are not used for the decision -- GitHub Actions reads the `proceed`
output. A non-zero exit means the gate itself failed.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd


def sessions_this_month(today: pd.Timestamp) -> pd.DatetimeIndex:
    """NYSE sessions in `today`'s month, from the S&P's own price index.

    The index the model rebalances against is the authority on what a trading
    day is -- deriving it from a holiday library instead would let the two
    disagree, and the disagreement would only show up as a book dated to a day
    the panel has no bar for.
    """
    import yfinance as yf

    start = (today - pd.Timedelta(days=45)).date()
    end = (today + pd.Timedelta(days=1)).date()
    hist = yf.download("^GSPC", start=start, end=end, interval="1d",
                       progress=False, auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError("gate: could not fetch the ^GSPC calendar")
    idx = pd.DatetimeIndex(pd.to_datetime(hist.index)).tz_localize(None).normalize()
    return idx[(idx.year == today.year) & (idx.month == today.month)]


def decide(today: pd.Timestamp) -> dict:
    sess = sessions_this_month(today)
    if len(sess) == 0:
        return {"proceed": False, "reason": "no sessions in this month yet"}

    book_date = sess[0]
    if len(sess) < 2:
        return {"proceed": False, "book_date": str(book_date.date()),
                "reason": f"entry bar not reached yet (book date {book_date.date()}, "
                          "the panel needs the session after it to price entry)"}

    entry_bar = sess[1]
    # The entry bar's close has to be IN THE PAST. Equal is not enough: the job
    # runs at 21:30 UTC, after the 16:00 ET close, so a same-day entry bar is
    # complete -- but only because of when the workflow is scheduled. Comparing
    # dates keeps that assumption in one place instead of spread across the
    # cron expression and this file.
    if entry_bar.date() > today.date():
        return {"proceed": False, "book_date": str(book_date.date()),
                "reason": f"entry bar {entry_bar.date()} has not closed yet"}

    lag = (today.normalize() - book_date).days
    return {
        "proceed": True,
        "book_date": str(book_date.date()),
        "entry_bar": str(entry_bar.date()),
        "lag_days": lag,
        # 3 is edge_tracker_lib._LOGGED_LATE_DAYS. Mirrored rather than imported
        # so the gate stays runnable without the model's dependency tree.
        "will_backfill": lag > 3,
        "reason": f"book {book_date.date()}, entry bar {entry_bar.date()} closed, lag {lag}d",
    }


def main() -> int:
    today = pd.Timestamp(os.environ.get("GATE_TODAY") or dt.date.today())
    forced = os.environ.get("GATE_FORCE", "").strip() == "1"

    try:
        d = decide(today)
    except Exception as e:                                  # network, mostly
        print(f"gate: FAILED to decide: {e}", file=sys.stderr)
        return 1

    if forced and not d["proceed"]:
        print(f"gate: FORCED past '{d['reason']}'")
        d["proceed"] = True
        d.setdefault("book_date", "")
        d["forced"] = True

    print(f"gate: proceed={d['proceed']} — {d['reason']}")
    if d.get("will_backfill"):
        print(f"::warning::book {d.get('book_date')} is {d.get('lag_days')} calendar "
              "days old; the forward record will mark this row BACKFILLED "
              "(kept for provenance, never scored).")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"proceed={'true' if d['proceed'] else 'false'}\n")
            fh.write(f"book_date={d.get('book_date', '')}\n")
            fh.write(f"lag_days={d.get('lag_days', '')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
