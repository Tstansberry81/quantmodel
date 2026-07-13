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
MCAP_FLOOR = 2e9
CORR_CAP = 0.50
CORR_LOOKBACK = 126
REGIME_EXPO = 0.25
COST_BPS = 10.0
SIGNAL = {"accel": 1.0}
GROWTH_MIX = 0.75                   # default mirrors the product (EDGE_SPEC growth_mix)
GROWTH_THRESH = 0.15               # YoY revenue-growth bar defining a "growth" name

SNAPSHOT_PATH = os.path.join("data", "cache", "edge_tracker.json")

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

# --- upgrade lineage --------------------------------------------------------
# Each tracked forward record carries the DELTAS vs the shipped production book,
# so as validated research upgrades accumulate over sessions we can see exactly
# what each candidate sleeve changes. "paper-upgraded" = production + every
# upgrade validated this campaign (currently: continuous regime + GP/assets gate).
UPGRADES = {
    "product": [],
    f"paper-n{PAPER_N}": [f"basket size {PAPER_N} (vs {N})"],
    "paper-upgraded": [
        "continuous 200dMA regime — evaluate the de-risk rule DAILY instead of "
        "freezing it per rebalance; cuts fast-crash drawdown (COVID-2020 ~-34%->-25%) "
        "at tied Sharpe (rigor 2026-07-13: DD win concentrated in fast V-crashes, "
        "mild first-half whipsaw, DSR pass)",
        "GP/assets quality gate — drop the bottom 40% by gross profitability "
        "(Novy-Marx) before the accel pick; shallower DD + higher Sortino at n=10 "
        "(stage-2 rigor 2026-07-13)",
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
            "price": _latest_price(data, ck),
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
            "5Y": round(5 * ppy), "10Y": round(10 * ppy),
            "20Y": round(20 * ppy), "MAX": n_rebals}
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


def _snap_config(s):
    """Which forward record a snapshot belongs to. Legacy entries (written before
    the paper-track split) carry no tag and are the product record."""
    return s.get("config", "product")


def _persist_snapshots(pan, data, mix):
    """Append the current rebalance's book to the snapshot file for BOTH forward
    records — the product (n=N) and the n=PAPER_N research candidate — the first
    time each (config, book_date) pair is seen, so both accrue genuinely forward
    over real calendar time. Returns the full snapshot list (newest last)."""
    snaps = _read_snapshots()
    i = pan.T - 1
    book_date = str(pan.bdates[i].date())
    changed = False
    for cfg, nn in (("product", N), (f"paper-n{PAPER_N}", PAPER_N)):
        if any(_snap_config(s) == cfg and s.get("book_date") == book_date for s in snaps):
            continue
        book, _bd = _current_book(pan, data, nn, mix)
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
                "signal": "acceleration + continuous 200dMA regime + GP/assets quality gate",
                "book_date": book_date,
                "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tickers": gp_tickers,
                "upgrades": UPGRADES.get("paper-upgraded", []),
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
        window = window if window in ("1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "MAX") else "MAX"
        n = int(n) if 5 <= int(n) <= 10 else N
        mix = round(float(mix), 2)
        if mix not in (0.0, 0.25, 0.5, 0.75, 1.0):
            mix = GROWTH_MIX
        pan = E.load_edge_panel(hold=hold)
        data = engine.load_bt_data()
        current_book, book_date = _current_book(pan, data, n, mix)
        full_log = _paper_log(pan, data, hold, n, mix)

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
