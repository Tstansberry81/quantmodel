"""The Edge Tracker -- live paper-portfolio backend.

Surfaces the CURRENT Edge book (today's full-spec picks) plus a running
paper-trading log seeded from backtest history, so the model is validated
forward, out-of-sample, in real time.

Read-only against edge_lib (imported as E) and qmodel.engine; the only thing
this module *writes* is a small snapshot file at data/cache/edge_tracker.json,
which accrues one entry per distinct rebalance book seen over real calendar
time. Nothing here mutates the backtest or its caches.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import numpy as np

import edge_lib as E
from qmodel import engine

# full-spec book parameters (must mirror EDGE_SPEC so the tracker == the product)
HOLD = 42
N = 20
MCAP_FLOOR = 2e9
CORR_CAP = 0.50
CORR_LOOKBACK = 126
REGIME_EXPO = 0.25
COST_BPS = 10.0
SIGNAL = {"accel": 1.0}
GROWTH_MIX = 0.75                   # default mirrors the product (EDGE_SPEC growth_mix)
GROWTH_THRESH = 0.15               # YoY revenue-growth bar defining a "growth" name

SNAPSHOT_PATH = os.path.join("data", "cache", "edge_tracker.json")


def _meta(data, ck):
    m = data.get(ck, {}).get("meta", {})
    return (m.get("ticker", ck), m.get("name", ck),
            m.get("sector", "Unknown"))


def _current_book(pan, data, n, mix):
    """The latest rebalance's full-spec picks = the live portfolio.

    Equal-weight (1/n) book honoring the growth mix (K=round(mix*n) names from the
    >=GROWTH_THRESH revenue-growth pool, rest pure acceleration); we also surface
    the raw acceleration signal so the reader can see the conviction ordering."""
    i = pan.T - 1
    df = pan.panels[i]
    floored = df[df["pit_mcap"] >= MCAP_FLOOR] if MCAP_FLOOR else df
    cks = E._blend_select(floored, pan.bdates[i], n, SIGNAL, CORR_CAP,
                          CORR_LOOKBACK, mix, GROWTH_THRESH)
    w = 1.0 / len(cks) if cks else 0.0
    accel = df.set_index("company_key")["accel"]
    book = []
    for ck in cks:
        tk, nm, sec = _meta(data, ck)
        book.append({
            "ticker": tk, "name": nm, "sector": sec, "weight": w,
            "accel": float(accel.get(ck, float("nan"))),
        })
    # show highest-conviction (accel) first
    book.sort(key=lambda r: (r["accel"] if r["accel"] == r["accel"] else -1e9),
              reverse=True)
    return book, str(pan.bdates[i].date())


def _paper_log(pan, data, hold, n, mix):
    """Seed the FULL log from backtest history: every rebalance is a closed paper
    trade with its realized NET Edge return vs the S&P (the most recent rebalance
    is the still-OPEN trade). Each entry also carries the actual basket held that
    rebalance (ticker + each name's own forward return). Returns the full list,
    newest last; the caller slices it to the requested window."""
    bdates, gross, net, turn, spxf, ndxf, holds = E._edge_full(
        hold, n, MCAP_FLOOR, CORR_CAP, CORR_LOOKBACK, REGIME_EXPO, COST_BPS,
        tuple(sorted(SIGNAL.items())), mix, GROWTH_THRESH)
    T = len(net)
    log = []
    for i in range(T):
        opened = bdates[i]
        # close date ~ hold trading days later (approx via 7/5 calendar scaling)
        closes = opened + np.timedelta64(int(round(hold * 7 / 5)), "D")
        edge_ret = float(net[i])
        sp_ret = float(spxf[i]) if spxf[i] == spxf[i] else 0.0
        status = "OPEN" if i == T - 1 else "CLOSED"
        # the basket held this rebalance + each name's own forward return
        fwd = pan.panels[i].set_index("company_key")["fwd_ret"]
        holdings = []
        for ck in holds[i]:
            tk, nm, _sec = _meta(data, ck)
            r = float(fwd.get(ck, float("nan")))
            holdings.append({"ticker": tk, "name": nm,
                             "ret": (r if r == r else None)})   # NaN -> None
        # best performer first; names with no return sort last
        holdings.sort(key=lambda h: (h["ret"] if h["ret"] is not None else -1e9),
                      reverse=True)
        log.append({
            "opened": str(opened.date()),
            "closes": str(np.datetime64(closes, "D")),
            "status": status,
            "edge_ret": edge_ret,
            "sp_ret": sp_ret,
            "excess": edge_ret - sp_ret,
            "holdings": holdings,
        })
    return log


def _log_stats(log):
    """Hit-rate / average-excess stats over the CLOSED trades in a (windowed) log."""
    closed = [r for r in log if r["status"] == "CLOSED"]
    n = len(closed)
    wins = sum(1 for r in closed if r["excess"] > 0)
    return {
        "n_closed": n,
        "hit_rate": wins / n if n else 0.0,
        "avg_excess": (sum(r["excess"] for r in closed) / n) if n else 0.0,
        "avg_edge_ret": (sum(r["edge_ret"] for r in closed) / n) if n else 0.0,
        "avg_sp_ret": (sum(r["sp_ret"] for r in closed) / n) if n else 0.0,
    }


# window label -> rebalances, derived from rebalances-per-year (ppy = 252/hold),
# matching the backtest's window math so the two pages agree.
def _window_k(n_rebals, ppy, window):
    wmap = {"1Y": round(ppy), "2Y": round(2 * ppy), "3Y": round(3 * ppy),
            "5Y": round(5 * ppy), "MAX": n_rebals}
    return max(2, min(int(wmap.get(window, n_rebals)), n_rebals))


def _read_snapshots():
    """Load the forward-accruing snapshot list (newest last); [] if absent/bad."""
    try:
        if os.path.exists(SNAPSHOT_PATH):
            with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
                snaps = json.load(f)
            return snaps if isinstance(snaps, list) else []
    except (OSError, ValueError):
        pass
    return []


def _persist_snapshot(book_date, current_book):
    """Append the current book + date to the snapshot file the first time we see
    this rebalance date, so the tracker genuinely accrues forward over real
    calendar time. Returns the snapshot list (newest last)."""
    snaps = _read_snapshots()
    already = any(s.get("book_date") == book_date for s in snaps)
    if not already:
        snaps.append({
            "book_date": book_date,
            "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tickers": [r["ticker"] for r in current_book],
        })
        try:
            os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
            with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
                json.dump(snaps, f, indent=2)
        except OSError:
            pass
    return snaps


def tracker_state(hold: int = HOLD, window: str = "MAX", n: int = N,
                  mix: float = GROWTH_MIX) -> dict:
    """Assemble the tracker payload consumed by /api/edge_tracker.

    `hold` (21/42/63/126 = 1M/2M/3M/6M) sets the rebalance clock; `window`
    (1Y/2Y/5Y/MAX) trims the paper-log to that horizon; `n` (10/20) sets the
    basket size; `mix` (0/.25/.5/.75/1) sets the growth mix. All four mirror the
    backtest page so the tracker can be explored over the same clocks, windows,
    basket sizes, and growth mixes. The log + its hit-rate/excess stats reflect
    the chosen window; the current book and the forward-accruing snapshot record
    are always the live values."""
    try:
        hold = int(hold) if int(hold) in (21, 42, 63, 126) else HOLD
        window = window if window in ("1Y", "2Y", "3Y", "5Y", "MAX") else "MAX"
        n = int(n) if int(n) in (10, 20) else N
        mix = round(float(mix), 2)
        if mix not in (0.0, 0.25, 0.5, 0.75, 1.0):
            mix = GROWTH_MIX
        pan = E.load_edge_panel(hold=hold)
        data = engine._load_bt_data()
        current_book, book_date = _current_book(pan, data, n, mix)
        full_log = _paper_log(pan, data, hold, n, mix)

        # trim to the requested window (same math as the backtest: ppy = 252/hold)
        k = _window_k(len(full_log), pan.ppy, window)
        log = full_log[-k:]
        log_stats = _log_stats(log)

        # The forward record only accrues at the real product config (2M clock,
        # 20-name book, default growth mix); other clocks/sizes/mixes are
        # exploratory and must not write phantom snapshots into the live record.
        snaps = (_persist_snapshot(book_date, current_book)
                 if hold == HOLD and n == N and mix == GROWTH_MIX else _read_snapshots())

        stats = {
            "n_snapshots": len(snaps),
            "n_closed": log_stats["n_closed"],
            "hit_rate": log_stats["hit_rate"],
            "avg_excess": log_stats["avg_excess"],
            "avg_edge_ret": log_stats["avg_edge_ret"],
            "avg_sp_ret": log_stats["avg_sp_ret"],
            "first_snapshot": snaps[0]["book_date"] if snaps else book_date,
            "window": window,
            "n_total": len(full_log),
        }
        return {
            "ok": True,
            "book_date": book_date,
            "window": window,
            "current_book": current_book,
            "log": log,
            "stats": stats,
            "spec": {
                "hold_days": hold, "n": n, "mcap_floor_bn": MCAP_FLOOR / 1e9,
                "corr_cap": CORR_CAP, "regime_expo": REGIME_EXPO,
                "cost_bps": COST_BPS, "signal": "acceleration (3m-prior3m)",
                "growth_mix": mix, "growth_thresh": GROWTH_THRESH,
            },
        }
    except Exception as e:  # surface a clean error to the page
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import pprint
    st = tracker_state()
    print("ok:", st["ok"], "| book_date:", st.get("book_date"))
    print("current_book (first 3):")
    pprint.pprint(st["current_book"][:3])
    print("log (last 3):")
    pprint.pprint(st["log"][-3:])
    print("stats:")
    pprint.pprint(st["stats"])
