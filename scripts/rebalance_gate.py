"""Decide whether TODAY is the day to run the monthly rebalance.

WHY A GATE INSTEAD OF A CRON EXPRESSION
---------------------------------------
The model's clock is "the first TRADING day of every month" (edge_lib._rebal_grid).
Cron cannot express a trading day: the 1st is a weekend or a holiday about a
third of the time, and the offset from the 1st to the first session moves every
month. So the workflow wakes on every weekday of the first eight days and this
script decides whether to proceed.

WHEN IT FIRES: TWICE
--------------------
PUBLISH -- the month's first session, after its close. The selection is already
final: 12-1 momentum ends 21 sessions back (arr[pos-21]/arr[pos-251]), so nothing
in the ranking is waiting on today's bar. The names go out now and the forward
record logs the book on its OWN date, at zero lag.

PRICE -- the next session, after its close. The t+1 bar the book is bought at now
exists, so entry prices land, the live mark starts from that bar, and the
published book stops saying "pending".

Missing PUBLISH is survivable: the PRICE run does both at lag 1. The old failure
mode was missing both, every month, forever.

LATENESS
--------
edge_tracker_lib._logged_late marks a book written down more than 3 CALENDAR days
after its own date as BACKFILLED -- kept for provenance, never scored. Publishing
on the book date makes lag 0, which retires the Friday-book-date edge case that
the entry-bar-only schedule could not avoid.

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
    """Two firings a month, not one.

    PUBLISH (the month's first session, after its close): the selection is fully
    determined -- 12-1 momentum ends 21 sessions back, so nothing in the ranking
    is waiting on anything. The names go out now, and the forward record gets the
    book on its own date at ZERO lag, which is the strongest evidence it can
    carry. The entry price does not exist yet and is not claimed to.

    PRICE (the next session, after its close): the t+1 bar the book is bought at
    now exists. Refresh so entry prices land, the live mark starts from that bar,
    and the published book stops saying "pending".

    Missing PUBLISH is survivable -- the PRICE run does both, just at lag 1.
    Missing both is what the old manual process did every month.
    """
    sess = sessions_this_month(today)
    if len(sess) == 0:
        return {"proceed": False, "reason": "no sessions in this month yet"}

    book_date = sess[0]
    lag = (today.normalize() - book_date).days
    common = {"book_date": str(book_date.date()), "lag_days": lag,
              # 3 is edge_tracker_lib._LOGGED_LATE_DAYS. Mirrored rather than
              # imported so the gate runs without the model's dependency tree.
              "will_backfill": lag > 3}

    if today.normalize() == book_date:
        return {**common, "proceed": True, "phase": "publish", "entry_bar": "",
                "reason": f"book {book_date.date()} — publishing the selection, "
                          "entry priced at the next close"}

    if len(sess) >= 2 and today.normalize() == sess[1]:
        return {**common, "proceed": True, "phase": "price",
                "entry_bar": str(sess[1].date()),
                "reason": f"entry bar {sess[1].date()} closed — pricing the "
                          f"{book_date.date()} book"}

    return {**common, "proceed": False, "phase": "",
            "reason": f"not a rebalance session (book {book_date.date()}, "
                      f"entry bar {sess[1].date() if len(sess) >= 2 else 'pending'})"}


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
            fh.write(f"phase={d.get('phase', '')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
