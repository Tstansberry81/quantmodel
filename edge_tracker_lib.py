"""The Edge Tracker -- live paper-portfolio backend.

Surfaces the CURRENT Edge book (today's full-spec picks) plus a running
paper-trading log seeded from backtest history, so the model is validated
forward, out-of-sample, in real time.

Read-only against edge_lib (imported as E) and the edge_data layer; the only thing
this module *writes* is a small snapshot file at data/cache/edge_tracker.json,
which accrues one entry per distinct rebalance book seen over real calendar
time. Nothing here mutates the backtest or its caches.
"""
from __future__ import annotations
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import edge_lib as E
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)

# full-spec book parameters (must mirror EDGE_SPEC so the tracker == the product)
HOLD = 42
N = 10                              # product basket size (10-stock book)
PAPER_N = 7                         # research candidate (2026-07 signal-hunt campaign):
                                    # paper-tracked forward ALONGSIDE the product book
MCAP_FLOOR = 1e10        # $10B: momentum is a large-cap effect
CORR_CAP = None          # off: it fights the signal (-2.5pts)
SECTOR_CAP = 2           # max 2 names per sector (mirrors EDGE_SPEC)
CORR_LOOKBACK = 126
REGIME_EXPO = 0.25
CONTINUOUS_REGIME = True # 2026-07-29: judge the 200dMA daily, not frozen at rebalance
VOL_TARGET = 0.25        # scale exposure toward 25% annualized vol (de-lever only)
VOL_LOOKBACK = 21
COST_BPS = 10.0
SIGNAL = {"ret_12_1": 1.0}
_SIGNAL_LABEL = "12-1 momentum (12-month return, skipping the last month)"

# --- STATUS OF THE SHIPPED SIGNAL -------------------------------------------
# 2026-07-26: acceleration was falsified on survivorship-free Sharadar data
# (21,946 tickers, 15,634 delisted, back to 1998). Three independent lenses:
#   * cross-section  -- top-10 tail +0.27%/rebalance at t=0.4, positive in only
#                       45% of rebalances: it does not rank the cross-section.
#   * out-of-sample  -- 8.3% CAGR vs the S&P's 14.8% (-6.4%/yr), max drawdown
#                       -52.8% vs -22.1%, over 2012-2026.
#   * forward record -- the live paper book's last two closed rebalances lost.
# The old +7%/yr excess came from a panel with 111 delisted names, ALL of which
# died in 2024 or later -- i.e. the backtest could not hold a stock through its
# death at any point before then. The tracker keeps accruing so the forward
# record stays honest and unbroken, but the page must not present this as a
# working product. Replacement is not chosen yet: momentum ranks better but has
# not yet beaten the index on a RISK-ADJUSTED basis out of sample.
MODEL_STATUS = {
    "state": "live",
    "since": "2026-07-29",
    "headline": "12-1 momentum, large caps, regime-gated, volatility-targeted.",
    "detail": ("Acceleration was retired on 2026-07-27 after it was falsified on "
               "survivorship-free data. The replacement ranks large caps by 12-1 "
               "momentum, caps each sector at 2 names, judges the 200-day-MA "
               "regime daily, and scales exposure toward 25% annualized "
               "volatility. Backtest, net of 10bps, 1999-2026: 18.3%/yr vs the "
               "S&P's 8.6%, Sharpe 0.93 vs 0.53, max drawdown -30.8% vs -55.3%. "
               "That is a BACKTEST. The forward record below starts at these "
               "parameters and is the only out-of-sample evidence."),
}
GROWTH_MIX = 0.0                    # retired with accel (ranked NEGATIVELY)
GROWTH_THRESH = 0.15               # YoY revenue-growth bar defining a "growth" name

# WHERE THE FORWARD RECORD LIVES.
#
# This file is the ONLY out-of-sample evidence the model has -- every other
# number on the site is a backtest. It is deliberately not in git (it accrues on
# whatever host is running) and not in the data bundle (the bundle is rebuilt
# from research and would overwrite it).
#
# That combination means the default path below is only durable if the host's
# filesystem is. On an ephemeral host it silently resets on every deploy, and a
# record that restarts without saying so is WORSE than no record: the page keeps
# presenting it as forward evidence. Point EDGE_TRACKER_PATH at a persistent
# volume in that case (render.yaml mounts one at /var/data).
SNAPSHOT_PATH = os.environ.get(
    "EDGE_TRACKER_PATH", os.path.join("data", "cache", "edge_tracker.json"))

# --- GP/assets quality-gate research sleeve (2026-07 fundamental campaign) -----
# Paper-tracked forward ALONGSIDE the product + n=7 books. Overlay = the accel
# pick AFTER routing out the lowest-gross-profitability names (Novy-Marx quality).
# Only the ~gate-0.40 n=10 config survived the stage-2 rigor (n=7 framings
# conflicted), so that is what we track. Depends on the fiscal.ai canonical
# fundamentals cache; if that file is absent (e.g. the live Render deploy), the
# record is simply skipped so the product tracker is unaffected.
GP_FUND_PKL = os.path.join("data", "cache", "fund_canonical.pkl")
GP_GATE_FRAC = 0.40                 # drop the bottom 40% by GP/assets
GP_GATE_N = 10                      # basket size that passed rigor
_FUND_CACHE = None

# --- PEAD earnings-confirmation gate (round-5 earnings dynamics) -------------
# Route out momentum names whose MOST RECENT earnings surprise was negative
# (time-series SUE<0, Foster-Olsen-Shevlin). Improves the book on the 2016+
# coverage window (modest, 2nd-half-loaded, sub-significant -> paper-track, not
# production). Needs the quarterly-EPS cache; skipped if absent.
PEAD_QEPS_PKL = os.path.join("data", "cache", "quarterly_eps.pkl")
PEAD_N = 10
_QEPS_CACHE = None

# --- upgrade lineage --------------------------------------------------------
# Each tracked forward record carries the DELTAS vs the shipped production book,
# so as validated research upgrades accumulate over sessions we can see exactly
# what each candidate sleeve changes. "paper-upgraded" = production + every
# upgrade validated this campaign (currently: continuous regime + GP/assets gate).
UPGRADES = {
    "product": [],
    f"paper-n{PAPER_N}": [f"basket size {PAPER_N} (vs {N})"],
    "paper-upgraded": [
        "GP/assets quality gate — drop the bottom 40% by gross profitability "
        "(Novy-Marx) before the momentum pick; shallower DD + higher Sortino at n=10 "
        "(stage-2 rigor 2026-07-13)",
        "[the continuous 200dMA regime this sleeve pioneered was PROMOTED TO "
        "PRODUCTION 2026-07-13, so it is no longer a delta vs the shipped book]",
    ],
    "paper-pead": [
        "PEAD earnings-confirmation gate — drop momentum names whose most recent "
        "earnings surprise was negative (time-series SUE<0, Foster-Olsen-Shevlin, "
        "reported quarterly EPS only); +CAGR and +Sharpe on the 2016+ coverage "
        "window at tied/better DD (n=10). Modest, 2nd-half-loaded, sub-significant "
        "(NW-t<2, ~10yr window) — paper-track candidate, not production.",
    ],
}

# --- exposure-rule candidates tested but NOT (yet) adopted -------------------
# Decision trail for the market-timing rules from the round-3/4 research, kept so
# the lineage is complete even though these are exposure rules (they change the
# book's EXPOSURE, not its holdings, so they can't be tracked as ticker snapshots).
# The paper-upgraded sleeve uses continuous-200dMA (the robust-drawdown choice).
EXPOSURE_CANDIDATES = {
    "continuous-200dMA": {
        "status": "ADOPTED — paper-upgraded sleeve",
        "rule": "evaluate the 200dMA de-risk DAILY instead of freezing it per rebalance",
        "result_n10": "18.5% / 0.91 / -26%  (vs product 18.9 / 0.90 / -34)",
        "note": "robust drawdown reducer — helps or ties in EVERY crisis (GFC/COVID/2022); "
                "tied Sharpe, small CAGR give-up. Rigor 2026-07-13: DSR pass, NW-t ~1.85 (n=10).",
    },
    "panic-state (Daniel-Moskowitz 2016 JFE)": {
        "status": "TESTED 2026-07-13, NOT adopted — logged alternative",
        "rule": "de-risk to 0.25 ONLY when S&P < 200dMA AND realized vol > its rolling median "
                "(the targeted crash state, vs the blanket 200dMA)",
        "result_n10": "19.6% / 0.94 / -28%  (vs product 18.9 / 0.90 / -34); NW-t +2.12 sig, DSR pass",
        "note": "BEST Sharpe/return of the campaign's timing rules AND excess-vs-S&P stays "
                "significant — BUT the drawdown protection is COVID-2020-specific: it was "
                "WORSE than production in the slow GFC (-17->-22) and 2022 (-18->-21) bears "
                "(stays invested in low-vol grinding bears). A return/Sharpe enhancer, not "
                "robust crash insurance. Revisit if the priority shifts from drawdown to Sharpe.",
    },
}


def _load_fund():
    """Lazy-load the canonical fundamentals pickle ({}=absent, cached once)."""
    global _FUND_CACHE
    if _FUND_CACHE is None:
        try:
            with open(GP_FUND_PKL, "rb") as f:
                _FUND_CACHE = pickle.load(f)
        except Exception:
            _FUND_CACHE = {}
    return _FUND_CACHE


def _gp_assets_asof(fd, when: pd.Timestamp) -> float:
    """GP/assets from the latest annual report filed on/before `when` (PIT-safe)."""
    idx = fd.index.values.astype("datetime64[ns]")
    pos = int(np.searchsorted(idx, np.datetime64(when, "ns"), side="right")) - 1
    if pos < 0:
        return float("nan")
    row = fd.iloc[pos]
    def num(k):
        try:
            return float(row.get(k))
        except Exception:
            return float("nan")
    gp, ta, rev, cogs = num("gross_profit"), num("total_assets"), num("revenue"), num("cogs")
    if not (gp == gp) and rev == rev and cogs == cogs:      # fall back to rev - cogs
        gp = rev - cogs
    return gp / ta if (gp == gp and ta == ta and ta) else float("nan")


def _gp_gate_tickers(pan, data, n=GP_GATE_N, frac=GP_GATE_FRAC):
    """Latest-rebalance GP-gated book tickers, or None if fundamentals are absent.
    Cheap: only the current cross-section is scored (no full-panel attach)."""
    fund = _load_fund()
    if not fund:
        return None
    i = pan.T - 1
    df = pan.panels[i]; bd = pd.Timestamp(pan.bdates[i])
    d = (df[df["pit_mcap"] >= MCAP_FLOOR] if MCAP_FLOOR else df).copy()
    d["_gpa"] = [(_gp_assets_asof(fund[ck], bd) if ck in fund else float("nan"))
                 for ck in d["company_key"]]
    thr = pd.to_numeric(d["_gpa"], errors="coerce").quantile(frac)
    keep = (d["_gpa"] >= thr) | d["_gpa"].isna()            # never exclude on missing data
    d = d[keep]
    cks = E._blend_select(d, pan.bdates[i], n, SIGNAL, CORR_CAP,
                          CORR_LOOKBACK, GROWTH_MIX, GROWTH_THRESH)
    return [_meta(data, ck)[0] for ck in cks]


def _load_qeps():
    """Lazy-load the quarterly-EPS pickle ({}=absent, cached once)."""
    global _QEPS_CACHE
    if _QEPS_CACHE is None:
        try:
            with open(PEAD_QEPS_PKL, "rb") as f:
                _QEPS_CACHE = pickle.load(f)
        except Exception:
            _QEPS_CACHE = {}
    return _QEPS_CACHE


def _sue_asof(df, when: pd.Timestamp) -> float:
    """Most-recent time-series SUE announced on/before `when` (PIT-safe), or NaN."""
    eps = pd.to_numeric(df["eps"], errors="coerce")
    d = eps - eps.shift(4)                                   # seasonal (YoY quarter) diff
    sue = (d / d.rolling(8, min_periods=4).std()).dropna()
    if len(sue) == 0:
        return float("nan")
    idx = sue.index.values.astype("datetime64[ns]")
    pos = int(np.searchsorted(idx, np.datetime64(when, "ns"), side="right")) - 1
    return float(sue.values[pos]) if pos >= 0 else float("nan")


def _pead_gate_tickers(pan, data, n=PEAD_N):
    """Latest book after routing out names whose most-recent earnings surprise was
    negative (SUE<0). None if the quarterly-EPS cache is absent."""
    qeps = _load_qeps()
    if not qeps:
        return None
    i = pan.T - 1
    df = pan.panels[i]; bd = pd.Timestamp(pan.bdates[i])
    d = (df[df["pit_mcap"] >= MCAP_FLOOR] if MCAP_FLOOR else df).copy()
    d["_sue"] = [(_sue_asof(qeps[ck], bd) if ck in qeps else float("nan"))
                 for ck in d["company_key"]]
    s = pd.to_numeric(d["_sue"], errors="coerce")
    d = d[(s >= 0) | s.isna()]                              # keep positive-surprise + unknown
    cks = E._blend_select(d, pan.bdates[i], n, SIGNAL, CORR_CAP,
                          CORR_LOOKBACK, GROWTH_MIX, GROWTH_THRESH)
    return [_meta(data, ck)[0] for ck in cks]


def _meta(data, ck):
    m = data.get(ck, {}).get("meta", {})
    return (m.get("ticker", ck), m.get("name", ck),
            m.get("sector", "Unknown"))


def _latest_price(data, ck):
    """Latest available close for a holding (from the quant-model price series)."""
    pr = data.get(ck, {}).get("prices")
    try:
        return round(float(pr.iloc[-1]), 2) if pr is not None and len(pr) else None
    except Exception:
        return None


def _current_book(pan, data, n, mix):
    """The STILL-OPEN position: the most recent scheduled rebalance's full-spec
    picks = what a live trader following the Edge is holding right now.

    Uses the panel's live rebalance (no realized forward return yet) when the
    data supports one; that is a full hold period fresher than the last
    backtest rebalance, which by construction must be old enough to have
    finished. Equal-weight (1/n) honoring the growth mix (K=round(mix*n) names
    from the >=GROWTH_THRESH revenue-growth pool, rest pure signal); we also
    surface the raw ranking signal so the reader can see the conviction
    ordering, and the mark-to-market return since the open."""
    live = pan.live_panel is not None and len(pan.live_panel) > 0
    df = pan.live_panel if live else pan.panels[pan.T - 1]
    bdate = pan.live_date if live else pan.bdates[pan.T - 1]
    floored = df[df["pit_mcap"] >= MCAP_FLOOR] if MCAP_FLOOR else df
    cks = E._blend_select(floored, bdate, n, SIGNAL, CORR_CAP,
                          CORR_LOOKBACK, mix, GROWTH_THRESH, require_fwd=not live,
                          sector_cap=SECTOR_CAP)
    w = 1.0 / len(cks) if cks else 0.0
    idx = df.set_index("company_key")
    # Read the column the model ACTUALLY ranks on, derived from SIGNAL, rather
    # than a hardcoded one. This page kept displaying and sorting by "accel"
    # after the model moved to 12-1 momentum, so the book was picked by one
    # number and presented ordered by a different, retired one.
    sig_col = max(SIGNAL, key=SIGNAL.get) if SIGNAL else "ret_12_1"
    sig = idx[sig_col] if sig_col in idx.columns else None
    todate = idx["fwd_todate"] if "fwd_todate" in idx.columns else None
    book = []
    for ck in cks:
        tk, nm, sec = _meta(data, ck)
        r = float(todate.get(ck, float("nan"))) if todate is not None else float("nan")
        s = float(sig.get(ck, float("nan"))) if sig is not None else float("nan")
        book.append({
            "ticker": tk, "name": nm, "sector": sec, "weight": w,
            "signal": s,
            "signal_col": sig_col,
            "price": _latest_price(data, ck),
            "ret_todate": (r if r == r else None),      # NaN -> None
        })
    # highest-conviction first, by the signal actually used
    book.sort(key=lambda r: (r["signal"] if r["signal"] == r["signal"] else -1e9),
              reverse=True)
    return book, str(pd.Timestamp(bdate).date()), cks, live


def _paper_log(pan, data, hold, n, mix, live_book, live_cks):
    """Seed the FULL log from backtest history: every backtest rebalance is a
    CLOSED paper trade with its realized NET Edge return vs the S&P. Each entry
    carries the actual basket held that rebalance (ticker + each name's own
    forward return).

    Every rebalance in the backtest panel has a *completed* hold window -- that
    is the panel's construction rule -- so none of them is still open. The open
    trade is the panel's live rebalance, appended last and marked to market at
    the latest close. Returns the full list, newest last; the caller slices it
    to the requested window."""
    # sector_cap MUST be passed: without it this log selected names the shipped
    # model would never hold (the current-book view passes it, so the two views
    # disagreed on what the book even is).
    bdates, gross, net, turn, spxf, ndxf, holds = E._edge_full(
        hold, n, MCAP_FLOOR, CORR_CAP, CORR_LOOKBACK, REGIME_EXPO, COST_BPS,
        tuple(sorted(SIGNAL.items())), mix, GROWTH_THRESH,
        sector_cap=SECTOR_CAP)
    T = len(net)
    log = []

    def _holdings(cks, ret_by_ck):
        out = []
        for ck in cks:
            tk, nm, _sec = _meta(data, ck)
            r = float(ret_by_ck.get(ck, float("nan")))
            out.append({"ticker": tk, "name": nm,
                        "ret": (r if r == r else None)})        # NaN -> None
        # best performer first; names with no return sort last
        out.sort(key=lambda h: (h["ret"] if h["ret"] is not None else -1e9), reverse=True)
        return out

    def _closes(opened):
        # close date ~ hold trading days later (approx via 7/5 calendar scaling)
        return str(np.datetime64(opened + np.timedelta64(int(round(hold * 7 / 5)), "D"), "D"))

    for i in range(T):
        edge_ret = float(net[i])
        sp_ret = float(spxf[i]) if spxf[i] == spxf[i] else 0.0
        log.append({
            "opened": str(bdates[i].date()),
            "closes": _closes(bdates[i]),
            "status": "CLOSED",
            "edge_ret": edge_ret,
            "sp_ret": sp_ret,
            "excess": edge_ret - sp_ret,
            "holdings": _holdings(holds[i], pan.panels[i].set_index("company_key")["fwd_ret"]),
        })

    if live_book and pan.live_date is not None and "fwd_todate" in pan.live_panel.columns:
        # The open trade: equal-weight mark-to-market since the rebalance, gross
        # of costs and before the regime overlay (nothing has settled yet).
        td = pan.live_panel.set_index("company_key")["fwd_todate"]
        rets = [float(td.get(ck, float("nan"))) for ck in live_cks]
        rets = [r for r in rets if r == r]
        edge_ret = float(np.mean(rets)) if rets else 0.0
        sp_ret = float(pan.live_spx_todate) if pan.live_spx_todate == pan.live_spx_todate else 0.0
        log.append({
            "opened": str(pd.Timestamp(pan.live_date).date()),
            "closes": _closes(np.datetime64(pd.Timestamp(pan.live_date), "ns")),
            "status": "OPEN",
            "edge_ret": edge_ret,
            "sp_ret": sp_ret,
            "excess": edge_ret - sp_ret,
            "mark_to_market": True,       # partial: the hold window hasn't finished
            "holdings": _holdings(live_cks, td),
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


# window label -> REBALANCES. Deliberately not edge_lib.window_k: that counts
# trading days for the backtest's daily curve, whereas the tracker's paper log
# has one row per rebalance. Only the WINDOWS list is shared, so the two pages
# can never offer a window the other doesn't know.
def _window_k(n_rebals, ppy, window):
    if window == "MAX" or window not in E.WINDOWS:
        return n_rebals
    return max(2, min(int(round(int(window[:-1]) * ppy)), n_rebals))


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


def _snap_config(s):
    """Which forward record a snapshot belongs to. Legacy entries (written before
    the paper-track split) carry no tag and are the product record."""
    return s.get("config", "product")


def _forward_log(snaps, full_log, cfg, book_date):
    """The GENUINELY FORWARD record: one entry per rebalance this system actually
    observed in real time, newest last.

    This is deliberately separate from `log`. That one is seeded from backtest
    history -- every rebalance since 1999 rendered as a "closed paper trade" --
    which is useful context but is NOT out-of-sample evidence, because the rules
    were chosen knowing how those periods turned out. Presenting the two in one
    list let a 164-row backtest visually swamp the handful of rows that are the
    only real evidence, and made the record look far longer than it is.

    Each entry carries `logged_at`: the wall-clock time this book was first
    written down. That timestamp, not the book date, is what makes a row
    forward -- it proves the basket was recorded before the outcome was known.
    Realized returns are joined from the backtest log once a hold window has
    actually completed; until then the row is OPEN.
    """
    # the backtest log keys its rebalance date as "opened", not "book_date"
    by_date = {e.get("opened"): e for e in full_log}
    out = []
    for s in [x for x in snaps if _snap_config(x) == cfg]:
        bd = s.get("book_date")
        match = by_date.get(bd)
        # A rebalance is only CLOSED once a realized forward return exists for
        # it; the live book is still open by construction, and the matching log
        # row carries status OPEN in that case.
        is_open = match is None or match.get("status") != "CLOSED"
        closed = match or {}
        out.append({
            "book_date": bd,
            "closes": closed.get("closes"),
            "logged_at": s.get("logged_at"),
            "tickers": s.get("tickers", []),
            "n": s.get("n"),
            "status": "OPEN" if is_open else "CLOSED",
            "edge_ret": None if is_open else closed.get("edge_ret"),
            "sp_ret": None if is_open else closed.get("sp_ret"),
            "excess": None if is_open else closed.get("excess"),
            "holdings": None if is_open else closed.get("holdings"),
        })
    out.sort(key=lambda e: (e["book_date"] or ""))
    return out


def _forward_stats(flog):
    """Stats over the forward record ONLY. Kept apart from the backtest-seeded
    log's stats so a hit rate computed over 164 simulated rebalances can never
    be read as the forward record's."""
    closed = [e for e in flog if e["status"] == "CLOSED" and e.get("excess") is not None]
    n_open = sum(1 for e in flog if e["status"] == "OPEN")
    if not closed:
        return {"n_closed": 0, "n_open": n_open, "hit_rate": None,
                "avg_excess": None, "avg_edge_ret": None, "avg_sp_ret": None,
                "first_logged": (flog[0].get("logged_at") if flog else None)}
    ex = [e["excess"] for e in closed]
    return {
        "n_closed": len(closed),
        "n_open": n_open,
        "hit_rate": sum(1 for v in ex if v > 0) / len(ex),
        "avg_excess": sum(ex) / len(ex),
        "avg_edge_ret": sum(e["edge_ret"] for e in closed) / len(closed),
        "avg_sp_ret": sum(e["sp_ret"] for e in closed) / len(closed),
        "first_logged": flog[0].get("logged_at") if flog else None,
    }


def _persist_snapshots(pan, data, mix):
    """Append the current rebalance's book to the snapshot file for BOTH forward
    records — the product (n=N) and the n=PAPER_N research candidate — the first
    time each (config, book_date) pair is seen, so both accrue genuinely forward
    over real calendar time. Returns the full snapshot list (newest last)."""
    snaps = _read_snapshots()
    # Key the record on the LIVE rebalance (the position actually open now), not
    # the last backtest rebalance -- the backtest grid necessarily lags a full
    # hold period, and snapshotting that date would freeze the forward record
    # one rebalance behind reality.
    book_date = str(pd.Timestamp(pan.live_date if pan.live_date is not None
                                 else pan.bdates[pan.T - 1]).date())
    changed = False
    for cfg, nn in (("product", N), (f"paper-n{PAPER_N}", PAPER_N)):
        if any(_snap_config(s) == cfg and s.get("book_date") == book_date for s in snaps):
            continue
        book, _bd, _cks, _live = _current_book(pan, data, nn, mix)
        snaps.append({
            "config": cfg,
            "n": nn,
            "book_date": book_date,
            "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tickers": [r["ticker"] for r in book],
            "upgrades": UPGRADES.get(cfg, []),
        })
        changed = True
    # "paper-upgraded" = production + every research upgrade validated this campaign
    # (continuous regime + GP/assets gate). Holdings differ from production via the
    # GP gate (the continuous-regime upgrade is an exposure rule, tracked as a noted
    # delta, not a holdings change). Only persisted if the fundamentals cache exists
    # (e.g. skipped on the live Render deploy), so the product tracker is unaffected.
    if not any(_snap_config(s) == "paper-upgraded" and s.get("book_date") == book_date
               for s in snaps):
        gp_tickers = _gp_gate_tickers(pan, data)
        if gp_tickers:
            snaps.append({
                "config": "paper-upgraded",
                "n": GP_GATE_N,
                "gate_frac": GP_GATE_FRAC,
                "signal": "12-1 momentum + continuous 200dMA regime + GP/assets quality gate",
                "book_date": book_date,
                "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tickers": gp_tickers,
                "upgrades": UPGRADES.get("paper-upgraded", []),
            })
            changed = True
    # "paper-pead" = accel book after the PEAD earnings-confirmation gate (drop
    # names with a negative most-recent earnings surprise). Needs the quarterly-EPS
    # cache; skipped if absent so the product tracker is unaffected.
    if not any(_snap_config(s) == "paper-pead" and s.get("book_date") == book_date
               for s in snaps):
        pead_tickers = _pead_gate_tickers(pan, data)
        if pead_tickers:
            snaps.append({
                "config": "paper-pead",
                "n": PEAD_N,
                "signal": "12-1 momentum + PEAD earnings-confirmation gate (drop SUE<0)",
                "book_date": book_date,
                "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tickers": pead_tickers,
                "upgrades": UPGRADES.get("paper-pead", []),
            })
            changed = True
    if changed:
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
    (1Y/2Y/5Y/MAX) trims the paper-log to that horizon; `n` (5..10) sets the
    basket size; `mix` (0/.25/.5/.75/1) sets the growth mix. All four mirror the
    backtest page so the tracker can be explored over the same clocks, windows,
    basket sizes, and growth mixes. The log + its hit-rate/excess stats reflect
    the chosen window; the current book and the forward-accruing snapshot record
    are always the live values."""
    try:
        hold = int(hold) if int(hold) in (21, 42, 63, 126) else HOLD
        window = window if window in E.WINDOWS else "MAX"
        n = int(n) if 5 <= int(n) <= 10 else N
        mix = round(float(mix), 2)
        if mix not in (0.0, 0.25, 0.5, 0.75, 1.0):
            mix = GROWTH_MIX
        pan = E.load_edge_panel(hold=hold)
        data = engine.load_bt_data()
        current_book, book_date, live_cks, live = _current_book(pan, data, n, mix)
        full_log = _paper_log(pan, data, hold, n, mix, live, live_cks)

        # trim to the requested window (same math as the backtest: ppy = 252/hold)
        k = _window_k(len(full_log), pan.ppy, window)
        log = full_log[-k:]
        log_stats = _log_stats(log)

        # Forward records accrue only at the real product clock + growth mix
        # (exploratory clocks/mixes must not write phantom snapshots). At that
        # config BOTH records persist — the 10-stock product book AND the n=7
        # research candidate — whichever n the viewer happens to be exploring.
        snaps = (_persist_snapshots(pan, data, GROWTH_MIX)
                 if hold == HOLD and mix == GROWTH_MIX else _read_snapshots())
        # the stats card reflects the record matching the viewed basket size
        viewed_cfg = f"paper-n{PAPER_N}" if n == PAPER_N else "product"
        vsnaps = [s for s in snaps if _snap_config(s) == viewed_cfg]
        # built off full_log (not the window-trimmed `log`) so a short viewing
        # window can never silently truncate the forward record
        forward_log = _forward_log(snaps, full_log, viewed_cfg, book_date)
        forward_stats = _forward_stats(forward_log)

        stats = {
            "n_snapshots": len(vsnaps),
            "record": viewed_cfg,
            "n_closed": log_stats["n_closed"],
            "hit_rate": log_stats["hit_rate"],
            "avg_excess": log_stats["avg_excess"],
            "avg_edge_ret": log_stats["avg_edge_ret"],
            "avg_sp_ret": log_stats["avg_sp_ret"],
            "first_snapshot": vsnaps[0]["book_date"] if vsnaps else book_date,
            "window": window,
            "n_total": len(full_log),
            # the open position is live, not a finished trade
            "book_is_live": bool(live),
            "book_regime_on": bool(pan.live_regime_on),
            "book_exposure": 1.0 if pan.live_regime_on else REGIME_EXPO,
        }
        return {
            "ok": True,
            "book_date": book_date,
            "window": window,
            "current_book": current_book,
            # `log` is BACKTEST-SEEDED history; `forward_log` is the only
            # out-of-sample record. Kept as separate keys so the page cannot
            # accidentally present one as the other.
            "log": log,
            "forward_log": forward_log,
            "forward_stats": forward_stats,
            "stats": stats,
            "spec": {
                "hold_days": hold, "n": n, "mcap_floor_bn": MCAP_FLOOR / 1e9,
                "corr_cap": CORR_CAP, "regime_expo": REGIME_EXPO,
                "cost_bps": COST_BPS, "signal": _SIGNAL_LABEL,
                "growth_mix": mix, "growth_thresh": GROWTH_THRESH,
                "sector_cap": SECTOR_CAP,
            },
            "model_status": MODEL_STATUS,
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
