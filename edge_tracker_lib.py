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
import logging
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import edge_lib as E
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)
import forward_ledger        # scores the forward record from ledger + prices

# `log` is a local list inside _paper_log, so the module logger is _log.
_log = logging.getLogger(__name__)

# Full-spec book parameters. READ from EDGE_SPEC, not retyped beside it.
#
# These used to be a hand-maintained copy with "must mirror EDGE_SPEC" written
# above them, which is the same promise the panel-cache version constant used to
# make -- and it has already been broken here more than once (the tracker ran
# without the sector cap while the page said it had one). A mirror that is
# derived cannot drift; a mirror that is retyped drifts the first time someone
# edits one file and not the other. RESEARCH_RULES #7.
_S = E.EDGE_SPEC
HOLD = _S["hold"]
REBAL_MONTHS = _S.get("rebal_months")   # calendar grid: books dated the 1st
N = 10                              # product basket size (10-stock book)
PAPER_N = 7                         # research candidate (2026-07 signal-hunt campaign):
                                    # paper-tracked forward ALONGSIDE the product book
MCAP_FLOOR = _S["mcap_floor"]       # $10B: momentum is a large-cap effect
CORR_CAP = _S["corr_cap"]           # off: it fights the signal (-2.5pts)
SECTOR_CAP = _S["sector_cap"]       # max 2 names per sector
CORR_LOOKBACK = _S["corr_lookback"]
REGIME_EXPO = _S["regime_expo"]
CONTINUOUS_REGIME = _S["continuous_regime"]  # judge the 200dMA daily, not at rebalance
VOL_TARGET = _S["vol_target"]       # scale exposure toward 25% annualized vol
VOL_LOOKBACK = _S["vol_lookback"]
COST_BPS = _S["cost_bps"]
FCF_POSITIVE = _S["fcf_positive"]        # solvency screens (2026-07-30)
DEBT_EBITDA_MAX = _S["debt_ebitda_max"]
SIGNAL = dict(_S["signal"])
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
    "since": "2026-07-30",
    "headline": ("12-1 momentum on solvent large caps, regime-gated, "
                 "volatility-targeted, rebalanced the first trading day of every month."),
    "detail": ("Acceleration was retired on 2026-07-27 after it was falsified on "
               "survivorship-free data. The replacement ranks large caps by 12-1 "
               "momentum, caps each sector at 2 names, judges the 200-day-MA "
               "regime daily, and scales exposure toward 25% annualized "
               "volatility. Since 2026-07-30 it also requires positive free cash "
               "flow and debt under 4x EBITDA — the only fundamental screens that "
               "improved return and risk together, and worth more than de-levering "
               "would have been. Backtest, net of 10bps, 1999-2026: 17.1%/yr vs "
               "the S&P's 8.7%, max drawdown -28.6% vs -55.3%. "
               "CORRECTION (2026-07-30): this page previously showed 18.3%/yr, "
               "Sharpe 0.93 and a -30.8% drawdown. Those came from a rebalance "
               "grid that started on an arbitrary date, and sweeping that start "
               "date showed it was the luckiest of six tested — the same model on "
               "other start dates drew down -43% to -48%. The grid is now anchored "
               "to calendar month starts, and the numbers above are what that "
               "actually produces. The old ones were too good by accident. "
               "All of it is still a BACKTEST; the forward record below is the "
               "only out-of-sample evidence."),
}
GROWTH_MIX = _S["growth_mix"]       # retired with accel (ranked NEGATIVELY)
GROWTH_THRESH = _S["growth_thresh"]  # YoY revenue-growth bar defining a "growth" name

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
# The default is anchored to THIS FILE, not the working directory. It used to be
# the relative "data/cache/edge_tracker.json", which means the record you get
# depends on where you happened to launch from: start the app from any other
# directory and it silently creates an empty record and the page presents that
# as the forward evidence. Caught exactly that way -- a dev server launched from
# a parent directory reported 1 snapshot while 4 sat on disk. Production sets
# EDGE_TRACKER_PATH to an absolute path on the mounted disk, so this only ever
# bit local runs, which is precisely where it is hardest to notice.
def _spec_sig() -> str:
    """Short hash of the parameters that decide WHICH NAMES are picked.

    A book recorded under v4 and a book recorded under v5 for the same rebalance
    date are different portfolios, not one record. Keying snapshots on
    (config, book_date) alone meant the first one written won forever: the
    2026-07-01 entry captured the pre-solvency-screen book and no later entry
    could ever be added for that date, so the forward record permanently showed
    a basket the model had stopped producing.

    Only selection inputs are included. Overlays (vol target, regime) change the
    exposure, not the holdings, so they must NOT reset a record of what was held.
    """
    import hashlib
    parts = (tuple(sorted(SIGNAL.items())), MCAP_FLOOR, SECTOR_CAP, CORR_CAP,
             FCF_POSITIVE, DEBT_EBITDA_MAX, HOLD, REBAL_MONTHS, GROWTH_MIX,
             GROWTH_THRESH)
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:8]


SPEC_SIG = None   # computed lazily below, after the constants exist


SNAPSHOT_PATH = os.environ.get(
    "EDGE_TRACKER_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "cache", "edge_tracker.json"))

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


def _num(series, ck):
    """One float off a company-key-indexed Series, or None. NaN never reaches the
    payload: JSON has no NaN, and json.dumps emits a bare `NaN` token that every
    strict parser rejects -- including the one Vision's static page uses."""
    if series is None or ck not in series.index:
        return None
    try:
        v = float(series.get(ck))
    except (TypeError, ValueError):
        return None
    return v if v == v else None


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
    # E._candidates, not a hand-rolled market-cap filter. This line used to be
    # `df[df["pit_mcap"] >= MCAP_FLOOR]`, which silently skipped every screen the
    # backtest applies -- exactly how the live book once ran without the sector
    # cap while the page advertised one. Going through the shared helper means a
    # screen added to the model reaches the published book automatically.
    floored = E._candidates(df, MCAP_FLOOR, False, FCF_POSITIVE, DEBT_EBITDA_MAX)
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
    # Where each pick stands in the FIELD it was chosen from. Computed here
    # because `floored` -- the eligible universe on this date -- exists at this
    # point and nowhere downstream. Vision's per-stock write-up uses it to say
    # why the model bought a name using real numbers instead of prose: "top 1%
    # of 812 names that cleared the floor" is a fact, not a characterisation.
    n_elig = int(len(floored))
    # sig_all is compared positionally ((sig_all < s).sum()), so `floored`'s
    # default index is fine. mcap is looked up BY company_key, so it has to come
    # off `idx` -- reading it from `floored` indexed the RangeIndex instead and
    # every lookup missed, which silently dropped the market-cap clause from the
    # published sentence rather than erroring.
    sig_all = floored[sig_col] if sig_col in floored.columns else None
    mcap_all = idx["pit_mcap"] if "pit_mcap" in idx.columns else None
    # The solvency figures the published write-up quotes. Carried through here
    # so Vision states the company's OWN numbers rather than just asserting it
    # passed a screen -- a claim the reader can check beats one they can't.
    fcf_all = idx["fcf_margin"] if "fcf_margin" in idx.columns else None
    de_all = idx["debt_ebitda"] if "debt_ebitda" in idx.columns else None
    book = []
    for ck in cks:
        tk, nm, sec = _meta(data, ck)
        r = float(todate.get(ck, float("nan"))) if todate is not None else float("nan")
        s = float(sig.get(ck, float("nan"))) if sig is not None else float("nan")
        pct = None
        if sig_all is not None and s == s and n_elig:
            # share of the eligible field this name outranks, 0..1
            pct = float((sig_all < s).sum()) / n_elig
        mc = None
        if mcap_all is not None and ck in mcap_all.index:
            v = float(mcap_all.get(ck, float("nan")))
            mc = v if v == v else None
        book.append({
            "ticker": tk, "name": nm, "sector": sec, "weight": w,
            "signal": s,
            "signal_col": sig_col,
            "price": _latest_price(data, ck),
            "ret_todate": (r if r == r else None),      # NaN -> None
            "pctile": pct,                  # rank within the eligible universe
            "n_eligible": n_elig,           # how big that universe was
            "mcap": mc,                     # point-in-time market cap
            "fcf_margin": _num(fcf_all, ck),
            "debt_ebitda": _num(de_all, ck),
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
        sector_cap=SECTOR_CAP,
        rebal_months=E.clock_spec(hold)["rebal_months"],
        fcf_positive=FCF_POSITIVE, debt_ebitda_max=DEBT_EBITDA_MAX)
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

    # Exit dates are READ from the panel, not derived. Under the month-start grid
    # the gap between rebalances runs 15-23 trading days, so no fixed day-count
    # tiles it. The previous version approximated the exit as hold*7/5 calendar
    # days -- correct when the clock was a fixed trading-day stride, wrong the
    # moment rebalancing became month-anchored (30 July 2026). It dated the
    # 2026-07-01 book's exit to 07-30 when the model actually holds it to 08-03.
    _nd = getattr(pan, "next_dates", None)
    _rebal_months = E.clock_spec(hold)["rebal_months"]

    def _closes_at(i):
        """Exit for closed rebalance `i`: the bar the NEXT book is entered on."""
        if _nd is not None and i < len(_nd) and pd.notna(_nd[i]):
            return str(pd.Timestamp(_nd[i]).date())
        return None

    def _closes_scheduled(opened):
        """Exit for the OPEN book -- the next rebalance, which has not happened.

        SCHEDULED, not observed: the exchange calendar does not extend into the
        future, so a holiday on the 1st pushes the real date later. Weekends are
        skipped here; holidays cannot be. Callers must not present this as a
        settled date.
        """
        if not _rebal_months:
            return None
        t = pd.Timestamp(opened)
        nxt = t.year * 12 + t.month + 1
        while (nxt - E.ANCHOR_MONTH) % _rebal_months != 0:
            nxt += 1
        d = pd.Timestamp(year=(nxt - 1) // 12, month=(nxt - 1) % 12 + 1, day=1)
        # New Year's Day is the only fixed-date US market holiday that can fall on
        # a month's 1st, so it is the one holiday knowable in advance. Everything
        # else (Good Friday, a funeral closure) still makes this an estimate.
        if d.month == 1 and d.day == 1:
            d += pd.Timedelta(days=1)
        while d.weekday() >= 5:                       # Sat/Sun -> next Monday
            d += pd.Timedelta(days=1)
        return str(d.date())

    for i in range(T):
        edge_ret = float(net[i])
        sp_ret = float(spxf[i]) if spxf[i] == spxf[i] else 0.0
        log.append({
            "opened": str(bdates[i].date()),
            "closes": _closes_at(i),
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
        # ENTRY PENDING: the names are final but the t+1 bar they are bought at
        # has not closed, so every fwd_todate is NaN. The old `if rets else 0.0`
        # would render that as a flat 0.00% row sitting next to a real S&P move --
        # a return for a position nobody has entered yet. None is the truth, and
        # the page can say "prices at the next close" instead of showing a number.
        pending = bool(getattr(pan, "live_entry_pending", False)) or not rets
        edge_ret = float(np.mean(rets)) if rets else None
        _sp = pan.live_spx_todate
        sp_ret = float(_sp) if (_sp == _sp and not pending) else None
        log.append({
            "opened": str(pd.Timestamp(pan.live_date).date()),
            "closes": _closes_scheduled(pan.live_date),
            "closes_scheduled": True,     # next rebalance, not yet on the calendar
            "status": "OPEN",
            "edge_ret": edge_ret,
            "sp_ret": sp_ret,
            "excess": (None if (edge_ret is None or sp_ret is None)
                       else edge_ret - sp_ret),
            "mark_to_market": not pending,   # partial: the hold window hasn't finished
            "entry_px_pending": pending or None,
            "entry_date": (str(pd.Timestamp(pan.live_entry_date).date())
                           if getattr(pan, "live_entry_date", None) is not None else None),
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
            if not isinstance(snaps, list):
                return []
            return _redate_held_over_august(snaps)
    except (OSError, ValueError):
        pass
    return []


# One-shot repair, 2026-09-01. Safe to delete once it has run in production.
#
# The 2026-08-03 rebalance never happened -- nothing automated the data refresh,
# so the panel froze and the 2026-07-01 book stayed open through all of August.
# The forward record therefore had no August row at all, while the July row sat
# at BACKFILLED (written down 29 days into its own window).
#
# The basket itself, though, was written down on 2026-07-30 -- BEFORE the August
# window opened. Held over an August window it is genuine forward evidence, and
# the only such evidence this record has. So the row is re-dated to the rebalance
# it actually spanned rather than left on a date whose window it missed.
#
# What it is NOT: a book the model picked on 2026-08-03. The names were selected
# on 2026-07-01 signals. `held_over` says so, and _forward_log refuses to score
# it against the 08-03 backtest rebalance, whose basket is different.
_REDATE_FROM = "2026-07-01"
_REDATE_TO = "2026-08-03"


def _redate_held_over_august(snaps):
    """Move the still-open 2026-07-01 product book onto the August rebalance."""
    if any(_snap_config(s) == "product" and s.get("book_date") == _REDATE_TO
           for s in snaps):
        return snaps                       # already applied, or a real 08-03 book exists

    # The NEWEST 07-01 product row is the book that was actually held; an older
    # one for the same date is a superseded pre-v5 basket and stays put.
    candidates = [s for s in snaps if _snap_config(s) == "product"
                  and s.get("book_date") == _REDATE_FROM]
    if not candidates:
        return snaps
    row = max(candidates, key=lambda s: s.get("logged_at") or "")

    row["book_date"] = _REDATE_TO
    row["held_over"] = True
    row["held_over_from"] = _REDATE_FROM
    row["held_over_note"] = (
        "Selected on 2026-07-01 signals and written down 2026-07-30. The 2026-08-03 "
        "rebalance did not run, so this basket was the live position for the whole "
        "August window — forward evidence for that window, but not a book the model "
        "re-picked on 2026-08-03.")
    try:
        os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
        with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
            json.dump(snaps, f, indent=2)
        _log.info("forward record: re-dated the held-over book %s -> %s",
                  _REDATE_FROM, _REDATE_TO)
    except OSError:
        _log.warning("forward record: could not persist the re-date", exc_info=True)
    return snaps


def _snap_config(s):
    """Which forward record a snapshot belongs to. Legacy entries (written before
    the paper-track split) carry no tag and are the product record."""
    return s.get("config", "product")


def _score_basket_at(pan, data, book_date, tickers):
    """Realized equal-weight return of an ARBITRARY basket at rebalance `book_date`.

    Needed because a held-over book is not the book the model picked at that
    rebalance, so its return cannot be read off the backtest log row -- that row
    carries the re-picked basket. The panel holds every eligible name's realized
    forward return for the window, so the held basket can be scored over the
    SAME window on the SAME data, which is the only way the comparison is
    apples-to-apples.

    Returns (edge_ret, sp_ret) or None when the window has not closed yet.
    """
    try:
        want = {str(t).upper() for t in (tickers or [])}
        if not want:
            return None
        bd = pd.Timestamp(book_date)
        hits = [i for i, d in enumerate(pan.bdates) if pd.Timestamp(d) == bd]
        if not hits:
            return None                      # not a closed rebalance on this grid
        i = hits[0]
        fwd = pan.panels[i].set_index("company_key")["fwd_ret"]
        rets = []
        for ck in fwd.index:
            tk, _nm, _sec = _meta(data, ck)
            if str(tk).upper() in want:
                v = float(fwd.get(ck, float("nan")))
                if v == v:
                    rets.append(v)
        if not rets:
            return None
        sp = float(pan.spxf[i]) if pan.spxf[i] == pan.spxf[i] else 0.0
        return float(np.mean(rets)), sp, len(rets)
    except Exception:                        # never let scoring break the page
        _log.warning("held-over scoring failed for %s", book_date, exc_info=True)
        return None


def _forward_log(snaps, full_log, cfg, book_date, score_basket=None):
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
    # Book dates the CURRENT grid can still produce. A snapshot written under a
    # different rebalance convention has no date on this grid, so it can never be
    # matched and would sit at OPEN forever -- a row that looks like a live
    # position and is actually a fossil. On 2026-07-30 the clock moved from a
    # 42-day stride to calendar month starts, which stranded every book dated
    # mid-month. Say so instead of leaving it pending in perpetuity.
    known = set(by_date)
    out = []
    for s in [x for x in snaps if _snap_config(x) == cfg]:
        bd = s.get("book_date")
        match = by_date.get(bd)
        # A rebalance is only CLOSED once a realized forward return exists for
        # it; the live book is still open by construction, and the matching log
        # row carries status OPEN in that case.
        is_open = match is None or match.get("status") != "CLOSED"
        # Stranded means "recorded under a clock this model no longer runs" --
        # judged by the row's OWN clock, not by whether the backtest grid has a
        # closed row for it yet. The grid lags a full hold: the night a new book
        # publishes, the previous month's book is neither closed nor live there,
        # and the old test stranded it (2026-10-01: the 09-01 book vanished from
        # the ledger and August was marked straight through to October).
        clock = s.get("clock") or {}
        on_clock = clock.get("rebal_months") == E.clock_spec(HOLD)["rebal_months"]
        stranded = (match is None and known and bd is not None and bd < max(known)
                    and not on_clock)
        # SUPERSEDED: a newer entry exists for the same book date under a
        # different selection spec. The old row is kept -- it is a genuine record
        # of what was written down at the time -- but it is no longer the book,
        # and showing it as live would misrepresent what the model holds.
        sig = s.get("spec_sig")
        superseded = any(
            x is not s and _snap_config(x) == cfg and x.get("book_date") == bd
            and x.get("spec_sig") != sig
            and (x.get("logged_at") or "") > (s.get("logged_at") or "")
            for x in snaps)
        # BACKFILLED: written down materially AFTER the date it is dated, so part
        # or all of its "forward" window was already in the past when the basket
        # was chosen. This is not a presentation nicety -- without it the
        # 2026-07-01 v5 book (first logged 2026-07-30, by which point ~91% of its
        # 07-01 -> 08-03 window had elapsed) would flip to CLOSED on 3 August and
        # post a hindsight return inside the panel labelled "the only
        # out-of-sample evidence". Kept visible for provenance, never scored --
        # the same treatment STRANDED and SUPERSEDED already get.
        backfilled = _logged_late(bd, s.get("logged_at"))

        # HELD OVER: this basket was carried into a rebalance it was not picked
        # at, because the rebalance did not run. Joining the matched log row's
        # return would post the return of the basket the model WOULD have picked
        # -- a number for a portfolio nobody held, printed next to the tickers
        # that actually were. So the row is scored on its own names over the same
        # window instead, and says which it is.
        held_over = bool(s.get("held_over"))
        held_score = None
        if held_over and score_basket is not None and not is_open:
            held_score = score_basket(bd, s.get("tickers") or [])
        closed = match or {}
        if held_over:
            # Never inherit the re-picked basket's numbers, even if scoring the
            # held names failed -- None reads as "not scored", which is true.
            closed = dict(closed)
            closed["holdings"] = None
            if held_score:
                closed["edge_ret"], closed["sp_ret"], _n = held_score
                closed["excess"] = closed["edge_ret"] - closed["sp_ret"]
            else:
                closed["edge_ret"] = closed["sp_ret"] = closed["excess"] = None
        out.append({
            "book_date": bd,
            "closes": closed.get("closes"),
            "logged_at": s.get("logged_at"),
            "logged_lag_days": _logged_lag_days(bd, s.get("logged_at")),
            "tickers": s.get("tickers", []),
            "n": s.get("n"),
            "clock": s.get("clock"),
            "status": ("SUPERSEDED" if superseded else
                       "STRANDED" if stranded else
                       "BACKFILLED" if backfilled else
                       ("OPEN" if is_open else "CLOSED")),
            "spec_sig": sig,
            # Carried as a FLAG rather than a status. The row is genuinely
            # forward -- the basket was written down before this window opened --
            # so it belongs in the hit rate and the average excess, which a
            # separate status value would have quietly excluded it from. The flag
            # is what the page labels; the note says what it means.
            "held_over": held_over or None,
            "held_over_from": s.get("held_over_from"),
            "held_over_note": s.get("held_over_note"),
            "held_over_n_priced": (held_score[2] if held_score else None),
            "stranded_reason": (
                "replaced by a later book for the same date after a change to the "
                "selection rules — kept as a record of what was written down, but "
                "it is not what the model holds" if superseded else
                "recorded on a rebalance clock this model no longer runs, so no "
                "closing date exists for it" if stranded else
                "written down after the date it is dated, so its holding window was "
                "already under way — shown for provenance, never scored, because a "
                "return measured over a window that had already happened is a "
                "backtest, not forward evidence" if backfilled else None),
            # Backfilled rows are scored as strictly as stranded ones: never.
            "edge_ret": None if (is_open or backfilled) else closed.get("edge_ret"),
            "sp_ret": None if (is_open or backfilled) else closed.get("sp_ret"),
            "excess": None if (is_open or backfilled) else closed.get("excess"),
            "holdings": None if (is_open or backfilled) else closed.get("holdings"),
        })
    out.sort(key=lambda e: (e["book_date"] or ""))
    return out


def _committed_on(flog, current_book):
    """The date this exact book was first written down, per the forward record.

    The live mark's inception should be the day the book was COMMITTED TO, not
    the day the process noticed it. Those differ whenever the live state is
    lost, and the forward record is the durable copy: it is snapshotted to the
    persistent disk the moment a book is picked, so it survives the deploys that
    used to reset the mark.

    Matched on the exact ticker set rather than the book date, because a book
    date is reused when the rules change mid-period (the 2026-07-01 date carries
    both the v4 and v5 baskets) and those are different portfolios.
    """
    want = sorted(str(b["ticker"]).upper() for b in (current_book or [])
                  if b.get("ticker"))
    if not want:
        return None
    dates = [(r.get("logged_at") or "")[:10] for r in (flog or [])
             if r.get("logged_at")
             and sorted(str(t).upper() for t in (r.get("tickers") or [])) == want]
    return min(dates) if dates else None


def _logged_lag_days(book_date, logged_at):
    """Calendar days between the date a book is DATED and the wall-clock moment
    it was first written down. Zero (or negative) is what forward evidence looks
    like; a large positive number means the window was already running."""
    if not book_date or not logged_at:
        return None
    try:
        lg = pd.Timestamp(logged_at)
        if lg.tz is not None:
            lg = lg.tz_convert(None)
        return int((lg.normalize() - pd.Timestamp(book_date).normalize()).days)
    except Exception:
        return None


# Entry is at the close AFTER the book date, so a row logged that day or the next
# session is genuine. 3 calendar days covers a weekend without letting real
# hindsight through.
_LOGGED_LATE_DAYS = 3


def _logged_late(book_date, logged_at):
    lag = _logged_lag_days(book_date, logged_at)
    return lag is not None and lag > _LOGGED_LATE_DAYS


def _forward_stats(flog):
    """Stats over the forward record ONLY. Kept apart from the backtest-seeded
    log's stats so a hit rate computed over 164 simulated rebalances can never
    be read as the forward record's."""
    closed = [e for e in flog if e["status"] == "CLOSED" and e.get("excess") is not None]
    n_open = sum(1 for e in flog if e["status"] == "OPEN")
    # Stranded rows are neither open nor closed, so both counts skip them. Report
    # the number anyway: rows that silently vanish from every total are how a
    # record ends up looking shorter than it is with no explanation on the page.
    n_stranded = sum(1 for e in flog if e["status"] == "STRANDED")
    n_superseded = sum(1 for e in flog if e["status"] == "SUPERSEDED")
    n_backfilled = sum(1 for e in flog if e["status"] == "BACKFILLED")
    if not closed:
        return {"n_closed": 0, "n_open": n_open, "n_stranded": n_stranded,
                "n_superseded": n_superseded, "n_backfilled": n_backfilled,
                "hit_rate": None,
                "avg_excess": None, "avg_edge_ret": None, "avg_sp_ret": None,
                "first_logged": (flog[0].get("logged_at") if flog else None)}
    ex = [e["excess"] for e in closed]
    return {
        "n_closed": len(closed),
        "n_open": n_open,
        "n_stranded": n_stranded,
        "n_superseded": n_superseded,
        "n_backfilled": n_backfilled,
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
        sig = _spec_sig()
        # NO DEFAULT on spec_sig. `s.get("spec_sig", sig)` reads as "unknown means
        # unchanged", which is backwards: a legacy entry written before this field
        # existed compared EQUAL to the current spec, so the guard skipped writing
        # the new book and the page kept showing the pre-solvency-screen basket --
        # exactly the bug the field was added to fix. Absent means different.
        if any(_snap_config(s) == cfg and s.get("book_date") == book_date
               and s.get("spec_sig") == sig for s in snaps):
            continue
        book, _bd, _cks, _live = _current_book(pan, data, nn, mix)
        snaps.append({
            "config": cfg,
            "n": nn,
            "book_date": book_date,
            "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            # Stamp the rebalance clock this book was written under. Without it,
            # a future clock change leaves rows that cannot be matched and no way
            # to tell why -- which is exactly what the 2026-07-30 move to a
            # calendar grid did to the books dated on the old 42-day stride.
            "clock": {"hold": HOLD, "rebal_months": REBAL_MONTHS},
            # Which SELECTION spec produced this basket. Without it, one record
            # per (config, date) meant a spec change could never be recorded for
            # a date already logged -- the page kept showing a book the model no
            # longer picks.
            "spec_sig": _spec_sig(),
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
            # Stamp the rebalance clock this book was written under. Without it,
            # a future clock change leaves rows that cannot be matched and no way
            # to tell why -- which is exactly what the 2026-07-30 move to a
            # calendar grid did to the books dated on the old 42-day stride.
            "clock": {"hold": HOLD, "rebal_months": REBAL_MONTHS},
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
            # Stamp the rebalance clock this book was written under. Without it,
            # a future clock change leaves rows that cannot be matched and no way
            # to tell why -- which is exactly what the 2026-07-30 move to a
            # calendar grid did to the books dated on the old 42-day stride.
            "clock": {"hold": HOLD, "rebal_months": REBAL_MONTHS},
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


def finalize(payload: dict) -> dict:
    """Apply the forward ledger to a (possibly cached) tracker payload.

    Returns a NEW dict; the cached payload is never mutated. Every forward row
    is rescored from its own entry close with its own names (forward_ledger),
    so the open row, the live cards and each name's "since open" all come from
    one computation and refresh with prices rather than with deploys.
    """
    if not payload.get("ok"):
        return payload
    out = dict(payload)
    try:
        led = forward_ledger.score_live(payload.get("forward_log") or [])
    except Exception:                                        # noqa: BLE001
        _log.warning("forward ledger failed; serving unscored rows", exc_info=True)
        return out
    flog = led["rows"]
    out["forward_log"] = flog
    out["forward_stats"] = _forward_stats(flog)

    spec = payload.get("spec") or {}
    product = ((payload.get("stats") or {}).get("record") == "product"
               and spec.get("hold_days") == HOLD)
    live = led["live"] if product else {}
    out["live"] = live
    total = led.get("total", {}) if product else {}
    if total:
        # Names for the drill-down: the forward ledger only knows tickers.
        names = {h.get("ticker"): h.get("name")
                 for e in (payload.get("log") or []) for h in (e.get("holdings") or [])
                 if h.get("ticker")}
        names.update({b.get("ticker"): b.get("name")
                      for b in (payload.get("current_book") or []) if b.get("ticker")})
        for leg in total.get("legs") or []:
            leg["holdings"] = [dict(h, name=names.get(h.get("ticker")))
                               for h in leg.get("holdings") or []]
    out["total_return"] = total
    out["stats"] = dict(payload.get("stats") or {}, book_is_live=bool(live))

    # Entry pending, per the ledger: the newest scoreable book has not been
    # bought yet. The page says "entry at the <date> close" and shows no number.
    newest = next((r for r in reversed(flog)
                   if r.get("status") in ("PENDING", "OPEN", "CLOSED")), None)
    if newest is not None and newest.get("book_date") == payload.get("book_date"):
        out["entry_px_pending"] = newest.get("status") == "PENDING"
        out["entry_date"] = newest.get("entry_date") or payload.get("entry_date")

    # Each name's return since the held book's entry, when the book on the page
    # IS the held book. A pending book has no return yet: None, not 0.
    held = led["held"]
    rets = ({h["ticker"]: h["ret"] for h in (held.get("holdings") or [])}
            if held and held.get("book_date") == payload.get("book_date") else {})
    out["current_book"] = [dict(b, ret_todate=rets.get(b.get("ticker")))
                           for b in (payload.get("current_book") or [])]
    return out


def tracker_state(hold: int = HOLD, window: str = "MAX", n: int = N,
                  mix: float = GROWTH_MIX, ledger: bool = True) -> dict:
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
        pan = E.load_edge_panel(**E.clock_spec(hold))
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
        forward_log = _forward_log(
            snaps, full_log, viewed_cfg, book_date,
            score_basket=lambda bd, tks: _score_basket_at(pan, data, bd, tks))
        forward_stats = _forward_stats(forward_log)

        # The forward rows' returns and the live mark are NOT computed here.
        # This payload is cached per process; the ledger is applied on top of
        # it by finalize(), fresh, every time it is served. See forward_ledger.
        live = {}

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
        payload = {
            "ok": True,
            "book_date": book_date,
            # The book is published on its own date, before the close it is bought
            # at. While this is true the names are final and every return field is
            # null -- the page must say "entry at the <entry_date> close" rather
            # than render a 0% that looks like a flat day.
            "entry_px_pending": bool(getattr(pan, "live_entry_pending", False)),
            # What the PANEL (the Sharadar bundle) says. finalize() replaces
            # entry_px_pending with the ledger's answer for the page; this one
            # stays as the data-coverage check post_deploy asserts on.
            "panel_entry_px_pending": bool(getattr(pan, "live_entry_pending", False)),
            "entry_date": (str(pd.Timestamp(pan.live_entry_date).date())
                           if getattr(pan, "live_entry_date", None) is not None else None),
            "window": window,
            "current_book": current_book,
            # `log` is BACKTEST-SEEDED history; `forward_log` is the only
            # out-of-sample record. Kept as separate keys so the page cannot
            # accidentally present one as the other.
            "log": log,
            "forward_log": forward_log,
            "live": live,
            "forward_stats": forward_stats,
            "stats": stats,
            "spec": {
                "hold_days": hold, "n": n, "mcap_floor_bn": MCAP_FLOOR / 1e9,
                # The clock CONVENTION, not just its length. Omitting this left
                # the page reporting "hold_days: 21" for books that are dated the
                # first trading day of a month -- true but incomplete, and the
                # incompleteness is the interesting part. Derived from the hold
                # actually being viewed, so the exploratory clock buttons report
                # their own convention rather than the shipped one.
                "rebal_months": E.clock_spec(hold)["rebal_months"],
                "corr_cap": CORR_CAP, "regime_expo": REGIME_EXPO,
                "cost_bps": COST_BPS, "signal": _SIGNAL_LABEL,
                "sector_cap": SECTOR_CAP,
                "fcf_positive": FCF_POSITIVE,
                "debt_ebitda_max": DEBT_EBITDA_MAX,
                # The two EXPOSURE overlays are reported so the page can say they
                # exist, but `overlays_in_returns` is False on purpose: this
                # module's book and paper log come from E._edge_full, which is a
                # PER-REBALANCE path with no vol_target/continuous_regime
                # parameters at all (see its signature). Those overlays act on the
                # DAILY curve via E._edge_daily, which only the backtest page uses.
                # Reporting them without this flag made the tracker claim a 25%
                # vol target for returns computed without one -- the same
                # describe-a-model-that-is-not-running failure as the old
                # hardcoded "acceleration" label (RESEARCH_RULES #7).
                "continuous_regime": CONTINUOUS_REGIME,
                "vol_target": VOL_TARGET, "vol_lookback": VOL_LOOKBACK,
                "overlays_in_returns": False,
            },
            "model_status": MODEL_STATUS,
        }
        # `ledger`, not `finalize`: a parameter named after the function shadowed
        # it, and every default call (export_vision) died on "'bool' object is
        # not callable" while the page -- which passes False -- looked fine.
        return finalize(payload) if ledger else payload
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
