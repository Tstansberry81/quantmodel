"""The Edge -- shared harness for the SHORT-TERM TRADING model (Porter & Co).

It scans the Russell-1000-proxy universe and ranks on MARKET DATA ONLY
(short-horizon momentum / relative strength / acceleration), no fundamentals apart
from an optional revenue-growth tilt. Holding period ~30-60 days. Two proprietary
overlays get tested on top:
  * correlation cap (<=0.50) so the basket isn't eight versions of one bet,
  * a 200-day-MA regime switch that raises cash below the line.

All data is read through the self-contained edge_data layer (cached prices,
benchmarks, daily-return matrix). Signals are formed on close t but the trade
enters at close t+1 (a realistic one-day execution lag). Evaluated over
1Y / 2Y / 5Y / MAX windows.

No statsmodels in the venv; any stats are done by hand in numpy.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from functools import lru_cache
import numpy as np, pandas as pd
import hashlib
import logging
import os
import pathlib
import pickle
import time as _time
import config
import edge_data as D                # self-contained Edge data layer (no Slow Burn deps)
import pit_universe as PIT           # pluggable PIT Russell-1000 universe layer

log = logging.getLogger(__name__)

# Restrict each rebalance to REAL point-in-time Russell-1000 members (and price
# delisted names) when a real PIT source is present, instead of the survivorship-
# biased top-1000-by-mcap proxy. Defaults to whatever data is actually available:
# True iff pit_universe found a real source (Norgate or a user CSV). Override with
# the EDGE_USE_PIT_UNIVERSE env var ("1"/"0").
_env = os.environ.get("EDGE_USE_PIT_UNIVERSE")
USE_PIT_UNIVERSE = (_env == "1") if _env in ("0", "1") else PIT.is_real_pit()

# Delisting treatment. A name whose price series ends INSIDE the forward hold
# window is included, with fwd_ret truncated at its last trade and this haircut
# applied to that final price (a proxy for the delisting/OTC fade the data
# cannot see). 0.0 = truncate only, no extra penalty. None = EXCLUDE such names
# entirely.
#
# DEFAULT CHANGED 2026-07-29: None -> 0.0. Excluding them was LOOK-AHEAD -- the
# panel dropped a name on date d because it knew the company would stop trading
# within the next 42 days, which a real trader on date d does not know.
# Measured: 18 of 1,640 positions (1.1%) are affected, and every one is an
# ACQUISITION (Immunex->Amgen, BellSouth->AT&T, Genentech->Roche,
# Schering-Plough->Merck, Anheuser-Busch->InBev, Motorola Mobility->Google...),
# not a bankruptcy -- so the look-ahead was mostly skipping deal premiums and
# biased AGAINST the model. Removing it costs 0.18pts of CAGR (18.47 -> 18.29%),
# Sharpe 0.938 -> 0.934, drawdown unchanged. Cheap price for a number that is
# structurally honest rather than quietly forward-looking.
#
# NOTE this haircut only reaches the PER-REBALANCE path (fwd_ret). The daily
# curve the site reports comes from daily_return_matrix(), which forward-fills a
# delisted price -- so a dead name is frozen at 0% return there rather than
# marked down. Right for an acquisition (you hold cash at the deal price),
# generous for a bankruptcy by the final gap. Bounded by the same 1.1% of
# positions. See RESEARCH_RULES.md #4.
_dh = os.environ.get("EDGE_DELIST_HAIRCUT")
try:
    DELIST_HAIRCUT: float | None = float(_dh) if _dh not in (None, "") else 0.0
except ValueError:
    DELIST_HAIRCUT = 0.0

HOLD = 21                            # ~30 calendar days (a 1-month trading clock)
LB = 251                             # trailing window for signals / beta
PPY = 252.0 / HOLD
RF_PER = config.RISK_FREE_ANNUAL / PPY
N = 10                               # basket size (agents may vary)
UNIVERSE = 1000                      # Russell-1000 proxy: top-N by point-in-time mcap

# short-horizon signal columns the trading model ranks on (market data only)
SIGNAL_COLS = ["ret_21", "ret_63", "ret_126", "ret_12_1", "accel", "hi_252", "rs_63"]

# The single source of truth for backtest/tracker window labels. app.py,
# edge_tracker_lib, export_vision, and the templates' window buttons must all
# agree with this list; everything but "MAX" is "<years>Y".
WINDOWS = ("1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "MAX")
# windows shown in the per-window summary table (3Y is selector-only)
SUMMARY_WINDOWS = tuple(w for w in WINDOWS if w != "3Y")


def window_k(window: str, n_days: int) -> int:
    """Trading days covered by a window label, capped at the series length.
    MAX (or anything unrecognised) = the whole series. The backtest measures on
    the DAILY curve, so windows are counted in trading days here; the tracker's
    per-rebalance log does its own rebalance-count math off the same WINDOWS."""
    if window == "MAX" or window not in WINDOWS:
        return n_days
    return min(int(window[:-1]) * 252, n_days)


class EdgePanel:
    def __init__(self, panels, bdates, spxf, ndxf, ma200_on, mkt_daily, sectors, hold,
                 live_panel=None, live_date=None, live_regime_on=True,
                 live_spx_todate=float("nan")):
        self.panels = panels          # list[DataFrame] per rebalance
        self.bdates = bdates          # DatetimeIndex
        self.spxf = spxf              # S&P fwd return per period (over the FULL hold window)
        self.ndxf = ndxf              # Nasdaq fwd return per period (over the FULL hold window)
        self.ma200_on = ma200_on      # bool array: market above its 200-day MA at d
        self.mkt_daily = mkt_daily    # S&P daily returns (Series)
        self.sectors = sectors
        self.T = len(panels)
        self.hold = hold                          # holding period in trading days
        self.ppy = 252.0 / hold                   # rebalances per year (annualization)
        self.rf_per = config.RISK_FREE_ANNUAL / self.ppy   # per-period risk-free
        # ---- the still-open position (see load_edge_panel) --------------------
        # Deliberately kept OUT of `panels`: these rows have no realized forward
        # return, so the backtest must never see them. `fwd_todate` marks each
        # name to the latest close instead.
        self.live_panel = live_panel              # DataFrame | None
        self.live_date = live_date                # Timestamp | None
        self.live_regime_on = live_regime_on      # market above its 200dMA at live_date
        self.live_spx_todate = live_spx_todate    # S&P return since live_date


def _build_edge_panel(hold: int, universe: int, offset_days: int) -> EdgePanel:
    """Build the trading panel once. Each row = one candidate's short-horizon
    market-data signals + forward return. NO fundamentals.

    Call load_edge_panel() instead: it adds the memo + disk cache around this."""
    data = D.load_bt_data(); bm = D.benchmarks()
    spx = bm["SP500"].dropna(); mret_full = spx.pct_change()
    ndx = bm["NASDAQ"].dropna()
    ma200 = spx.rolling(200).mean()

    prep = {}
    for ck, blob in data.items():
        pr = blob.get("prices")
        if pr is None or len(pr) < LB + hold + 2:
            continue
        arr = np.asarray(pr.values, float)
        pidx = np.asarray(pr.index.values, "datetime64[ns]")
        rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:] / arr[:-1] - 1
        mret = np.asarray(mret_full.reindex(pr.index, method="ffill").values, float)
        fh = blob.get("fund_hist"); fidx = None; mcap = None; rgr = None
        funds = {}
        if fh is not None and not fh.empty:
            fidx = np.asarray(fh.index.values, "datetime64[ns]")
            if "calculated_market_cap" in fh.columns:
                mcap = np.asarray(fh["calculated_market_cap"].values, float)
            if "growth_revenue_1y" in fh.columns:
                rgr = np.asarray(fh["growth_revenue_1y"].values, float)
            # Any other fundamental the artifact carries rides along, so a new
            # ingest column becomes screenable without touching this loader.
            for c in fh.columns:
                if c not in ("calculated_market_cap", "growth_revenue_1y"):
                    funds[c] = np.asarray(fh[c].values, float)
        prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret,
                    "fidx": fidx, "mcap": mcap, "rgr": rgr, "funds": funds,
                    "sector": blob.get("meta", {}).get("sector", "Unknown")}

    def asof(fidx, varr, dlag):
        if fidx is None or varr is None:
            return np.nan
        p = int(np.searchsorted(fidx, dlag, side="right")) - 1
        return varr[p] if p >= 0 else np.nan

    lag = np.timedelta64(90, "D")
    monthly = spx.index[spx.index >= spx.index.min() + pd.Timedelta(days=400)]
    # offset_days shifts the whole rebalance grid (for staggered sleeves); 0 = default.
    monthly = monthly[offset_days:]
    rebal = list(monthly[:-hold][::hold])
    # The scheduled rebalance grid runs `hold` days past the backtest panel's
    # last date. That final entry is the position a live trader following the
    # Edge opened and is STILL HOLDING -- no realized forward return yet, which
    # is exactly why the backtest grid stops short of it. Build it here (same
    # prep, same rules) but keep it out of `panels`.
    sched = list(monthly[::hold])
    live_d = sched[-1] if (sched and (not rebal or sched[-1] != rebal[-1])) else None
    panels, bdates, spxf, ndxf, ma200_on = [], [], [], [], []
    live_rows, live_date, live_regime_on, live_spx_todate = None, None, True, np.nan
    use_pit = USE_PIT_UNIVERSE and PIT.is_real_pit()
    spx_end = np.datetime64(pd.Timestamp(spx.index[-1]), "ns")   # sample end, for delist detection

    for d in (rebal + ([live_d] if live_d is not None else [])):
        is_live = live_d is not None and d == live_d
        dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt - lag
        # market trailing returns for relative strength
        mpos = int(np.searchsorted(spx.index.values.astype("datetime64[ns]"), dt, side="right")) - 1
        m63 = float(spx.iloc[mpos] / spx.iloc[mpos - 63] - 1) if mpos >= 63 else 0.0
        # REAL PIT universe: restrict to actual Russell-1000 members on date d
        # (includes delisted names that were members then). Empty set => the
        # adapter can't answer this date, so fall back to the mcap proxy below.
        pit_set = PIT.pit_members(pd.Timestamp(d)) if use_pit else frozenset()
        cand = []
        for ck, P in prep.items():
            if pit_set and ck not in pit_set:
                continue
            pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
            if pos < LB:
                continue
            n_fwd = len(P["arr"]) - 1 - pos
            full_fwd = n_fwd >= hold + 1          # +1: enter next day (t+1)
            if is_live:
                # No forward window exists yet for the open position. Require the
                # name to still be trading at the rebalance (so a ticker that died
                # months ago can't be resurrected into the live book on stale
                # prices) and to have the t+1 bar the entry is priced at.
                if P["pidx"][-1] < dt - np.timedelta64(7, "D") or n_fwd < 1:
                    continue
                full_fwd = False
            elif not full_fwd:
                # Series ends inside the hold window. Normally skip (the stock
                # is unbuyable-through-death in this data); under the delist
                # stress knob include it as a truncated position -- but only if
                # the series truly dies before the sample does (otherwise it's
                # just the end of the data, not a delisting), and only if there
                # is still a t+1 bar to enter on.
                died = P["pidx"][-1] < spx_end - np.timedelta64(7, "D")
                if DELIST_HAIRCUT is None or not died or n_fwd < 1:
                    continue
            mc = asof(P["fidx"], P["mcap"], dlag)
            if mc is None or np.isnan(mc):
                continue
            cand.append((mc, ck, pos, full_fwd))
        if len(cand) < 50:
            continue
        cand.sort(key=lambda x: x[0], reverse=True)
        # With real PIT membership we keep ALL members (membership IS the
        # universe definition); only the proxy needs the top-N-by-mcap cut.
        if not pit_set:
            cand = cand[:universe]
        rows = []
        for mc, ck, pos, full_fwd in cand:
            P = prep[ck]; arr = P["arr"]; w = P["rets"][pos - LB + 1: pos + 1]
            wm = P["mret"][pos - LB + 1: pos + 1]
            sd = w.std(ddof=1)
            ok = ~np.isnan(w) & ~np.isnan(wm); beta = np.nan
            if ok.sum() > 60 and wm[ok].var() > 0:
                beta = float(np.cov(w[ok], wm[ok])[0, 1] / wm[ok].var())
            ret_63 = arr[pos] / arr[pos - 63] - 1
            row = {
                "company_key": ck, "sector": P["sector"], "pit_mcap": mc, "beta": beta,
                "ret_21": arr[pos] / arr[pos - 21] - 1,
                "ret_63": ret_63,
                "ret_126": arr[pos] / arr[pos - 126] - 1,
                # 12-1 momentum (Jegadeesh-Titman): the canonical formulation --
                # 12-month return SKIPPING the most recent month, because that
                # last month carries short-term reversal that works against it.
                # pos >= LB (251) so pos-251 is always in bounds.
                "ret_12_1": arr[pos - 21] / arr[pos - 251] - 1,
                "accel": ret_63 - (arr[pos - 63] / arr[pos - 126] - 1),
                "hi_252": arr[pos] / arr[pos - 251: pos + 1].max(),
                # The NAME's own trend (price / its 200-day MA). The market regime
                # gate cannot see this: in 2000 and 2021 the momentum names broke
                # their own 200dMA while the index stayed above its own.
                "px_ma200": arr[pos] / arr[pos - 199: pos + 1].mean(),
                "rs_63": ret_63 - m63,                  # relative strength vs market
                "vol_21": float(w[-21:].std(ddof=1) * np.sqrt(252)) if sd > 0 else np.nan,
                "rev_growth": asof(P["fidx"], P["rgr"], dlag),   # YoY rev growth (90d-lagged)
                # EXECUTION LAG: signal is formed on close t but the trade enters at
                # close t+1 (you can't compute the whole cross-section AND trade the
                # close it was computed from). Validated benign-to-favorable, so the
                # tradeable next-day entry is the honest default. On the LIVE
                # rebalance no window has finished yet; under the delist stress knob
                # a dying name is truncated at its last trade with the haircut
                # applied -- every branch enters at the same t+1 close.
                "fwd_ret": (arr[pos + 1 + hold] / arr[pos + 1] - 1) if full_fwd
                           else (np.nan if is_live
                                 else (arr[-1] * (1.0 + DELIST_HAIRCUT)) / arr[pos + 1] - 1),
            }
            # Fundamentals, read as-of the SAME lagged date as revenue growth,
            # so every fundamental in the row was public before the trade.
            for _c, _v in P["funds"].items():
                row[_c] = asof(P["fidx"], _v, dlag)
            if is_live:                       # mark the open position to the latest close
                row["fwd_todate"] = arr[-1] / arr[pos + 1] - 1
            rows.append(row)
        regime_on = bool(spx.iloc[mpos] > ma200.iloc[mpos]) if not np.isnan(ma200.iloc[mpos]) else True
        if is_live:
            live_rows = pd.DataFrame(rows)
            live_date = pd.Timestamp(d)
            live_regime_on = regime_on
            live_spx_todate = float(spx.iloc[-1] / spx.iloc[mpos] - 1)
            continue
        panels.append(pd.DataFrame(rows)); bdates.append(pd.Timestamp(d))
        # benchmark forward return over the FULL hold-day window (hold-aware!)
        spxf.append(_bench_fwd_hold(spx, d, hold))
        ndxf.append(_bench_fwd_hold(ndx, d, hold))
        ma200_on.append(regime_on)

    sectors = sorted({s for df in panels for s in df["sector"].unique() if str(s).lower() != "unknown"})
    return EdgePanel(panels, pd.DatetimeIndex(bdates), np.array(spxf), np.array(ndxf),
                     np.array(ma200_on, bool), mret_full, sectors, hold,
                     live_panel=live_rows, live_date=live_date,
                     live_regime_on=live_regime_on, live_spx_todate=live_spx_todate)


# ---- panel disk cache -------------------------------------------------------
# Building a panel costs ~16s on a fast laptop and 3-4x that on a small cloud
# instance -- enough, on a cold start, to blow through a reverse proxy's ~100s
# request timeout before the first backtest returns a byte. The panel is a pure
# function of (artifact, code, knobs), so it is precomputed at bundle time and
# shipped; the host unpickles instead of rebuilding.
#
# A stale panel would be the worst kind of bug: right-looking numbers from the
# wrong data, silently (RESEARCH_RULES #5 -- caches are part of the answer). So
# the cache is FINGERPRINTED and a miss rebuilds rather than guesses. The
# fingerprint hashes this whole source file, not a hand-maintained version
# constant: "bump the version when you change the builder" is a rule that gets
# forgotten exactly once and then lies forever. Hashing the file over-
# invalidates (an unrelated edit rebuilds too) -- the cheap direction to err.
def _panel_fingerprint(hold: int, universe: int, offset_days: int) -> str:
    h = hashlib.sha256()
    h.update(pathlib.Path(__file__).read_bytes())          # the builder itself
    art = config.ARTIFACT_DIR / "backtest_data.pkl"        # the data it reads
    if art.exists():
        st = art.stat()
        h.update(f"{st.st_size}".encode())
    meta = config.ARTIFACT_DIR / "meta.json"
    if meta.exists():
        h.update(meta.read_bytes())
    # module-level research knobs: NOT function args, so they must be hashed in
    # or flipping one would reuse the other variant's panel.
    h.update(repr((hold, universe, offset_days, LB,
                   USE_PIT_UNIVERSE, DELIST_HAIRCUT)).encode())
    return h.hexdigest()[:16]


# Where a REBUILT panel gets cached. Defaults to data/cache (right for local dev
# and for reading the panel shipped in the bundle), but on an ephemeral host that
# directory dies with the instance -- so a rebuild triggered by a fingerprint
# miss is paid again on EVERY cold start, not once.
#
# That is not hypothetical: the fingerprint hashes edge_lib.py, so any code
# change here invalidates the panel shipped in the bundle, and the bundle only
# moves when DATA_URL/DATA_VERSION are bumped by hand. Between a code deploy and
# a data re-upload the site is guaranteed to be in that state. Pointing this at
# the persistent disk makes the rebuild cost land once instead of forever.
PANEL_CACHE_DIR = pathlib.Path(
    os.environ.get("EDGE_PANEL_CACHE_DIR", str(config.CACHE_DIR)))


def _panel_cache_paths(hold: int, universe: int, offset_days: int):
    """(paths to try reading, path to write).

    Read and write targets differ on purpose. The bundled panel lives in
    data/cache and is read-only in effect (the next deploy overwrites it), while
    a REBUILT panel must land somewhere that survives a restart. Writing the
    rebuild back over the bundled path -- the obvious one-path version of this --
    would put it straight back on ephemeral storage, which is exactly the case
    this exists to fix.
    """
    name = f"panel_h{hold}_u{universe}_o{offset_days}.pkl"
    shipped = config.CACHE_DIR / name
    persist = PANEL_CACHE_DIR / name
    reads = [shipped] if shipped == persist else [shipped, persist]
    return reads, persist


@lru_cache(maxsize=10)
def load_edge_panel(hold: int = HOLD, universe: int = UNIVERSE, offset_days: int = 0) -> EdgePanel:
    """Memoized panel: RAM -> disk -> build. See _panel_fingerprint for why a
    disk hit is only trusted when the fingerprint matches exactly."""
    want = _panel_fingerprint(hold, universe, offset_days)
    reads, path = _panel_cache_paths(hold, universe, offset_days)
    for cand in reads:
        if not cand.exists():
            continue
        try:
            with open(cand, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("fingerprint") == want:
                return blob["panel"]
            log.info("panel cache %s is stale (data or code changed); rebuilding",
                     cand)
        except Exception:
            # A corrupt or cross-version pickle must never be fatal: the whole
            # point of this cache is speed, and rebuilding always works.
            log.warning("panel cache %s unreadable; rebuilding", cand, exc_info=True)

    t0 = _time.time()
    panel = _build_edge_panel(hold, universe, offset_days)
    log.info("panel rebuilt in %.0fs (hold=%s universe=%s)",
             _time.time() - t0, hold, universe)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"fingerprint": want, "panel": panel}, fh, protocol=4)
        tmp.replace(path)      # atomic: a reader never sees a half-written panel
        log.info("panel cached to %s", path)
    except Exception:
        log.warning("could not write panel cache %s", path, exc_info=True)
    return panel


# ---- ranking + selection ----------------------------------------------------
def _z(s):
    s = pd.to_numeric(s, errors="coerce"); mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else pd.Series(0.0, index=s.index)


def neutralize(comp: pd.Series, df: pd.DataFrame, sector: bool = False,
               beta: bool = False) -> pd.Series:
    """Strip sector and market-beta exposure out of a cross-sectional score.

    A raw momentum score is not a pure stock-selection bet: whatever sector is
    running leads the ranking, and high-beta names dominate in a rising tape, so
    the 'edge' is partly a leveraged sector bet. Both are removed here in score
    space (before selection), which is the standard treatment:

      sector -- subtract the sector mean, so a name competes against its OWN
                sector rather than against whatever sector is hot.
      beta   -- subtract the cross-sectional OLS fit on beta, leaving the part
                of the score that beta does not explain.

    Names with no beta keep their score rather than being dropped: absent data
    is not evidence of neutrality, and dropping them would quietly bias the
    universe toward names with a long enough history to estimate beta.

    Both are removed in ONE joint regression, not two passes. Demeaning by
    sector and then residualising on beta does NOT leave a score orthogonal to
    both: beta varies across sectors, so the beta step puts the sector tilts
    back (measured: group means return to ~1e-1 instead of ~1e-17). Regressing
    on sector dummies and beta together is orthogonal to both by construction."""
    s = pd.to_numeric(comp, errors="coerce")
    if not (sector or beta):
        return s
    parts = []
    if sector and "sector" in df.columns:
        parts.append(pd.get_dummies(df["sector"].astype(str).reindex(s.index),
                                    dtype=float))
    if beta and "beta" in df.columns:
        b = pd.to_numeric(df["beta"], errors="coerce").reindex(s.index)
        parts.append(b.fillna(b.mean()).to_frame("beta"))
    if not parts:
        return s
    X = pd.concat(parts, axis=1)
    X["_const"] = 1.0
    y = s.fillna(s.mean())
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    ok = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
    if int(ok.sum()) <= Xv.shape[1] + 5:      # too few names to fit safely
        return s
    # lstsq gives the least-norm solution, so the dummies+intercept collinearity
    # is harmless and no reference category has to be dropped by hand.
    coef, *_ = np.linalg.lstsq(Xv[ok], yv[ok], rcond=None)
    return pd.Series(yv - Xv @ coef, index=s.index)


def score(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Cross-sectional composite of short-horizon signals (BIC-on-short-clock).
    Default = pure 3-month momentum / relative strength."""
    weights = weights or {"ret_63": 1.0}
    comp = pd.Series(0.0, index=df.index)
    for col, wt in weights.items():
        if col in df.columns:
            comp = comp + wt * _z(df[col]).fillna(0.0)
    return comp


def _scorable(df: pd.DataFrame, require_fwd: bool) -> pd.DataFrame:
    """Rows we're allowed to pick from. The backtest requires a realized forward
    return (a name we couldn't have measured must not enter a basket); the LIVE
    book has no forward return by definition, so it opts out."""
    return (df.dropna(subset=["fwd_ret"]) if require_fwd else df).copy()


def select_topN(df: pd.DataFrame, n: int = N, weights: dict | None = None,
                require_fwd: bool = True) -> list:
    d = _scorable(df, require_fwd)
    d["_c"] = score(d, weights)
    return list(d.sort_values("_c", ascending=False).head(n)["company_key"])


def corr_cap_select(df, asof=None, n=N, weights=None, cap=0.50, lookback=126,
                    require_fwd=True, corr_mode="max", sector_neutral=False,
                    beta_neutral=False):
    """Greedy basket: walk names best-first, admitting a name only if it is
    sufficiently uncorrelated with what has already been admitted.

    `corr_mode` decides what "sufficiently" means:
      "max"  -- legacy: reject if the WORST single pair exceeds `cap`.
      "mean" -- SUMMATIVE: reject if the AVERAGE pairwise correlation against
                the whole basket exceeds `cap`. This is the diversification
                measure that actually matters: for an equal-weight book the
                portfolio variance falls with the MEAN pairwise correlation
                (var ~ 1/n + (n-1)/n * rho_bar), not with the single worst
                pair. Under "max" a name can sit at 0.49 against all nine
                existing holdings -- passing every pairwise test -- while being
                almost perfectly redundant with the basket as a whole.
                (Mean <= cap is the normalised form of sum <= cap * len(chosen),
                so the constraint stays comparable as the basket grows.)

    `asof` (the rebalance date) is REQUIRED for a point-in-time window -- the
    correlation is judged on the `lookback` trading days ending at `asof`, never
    future data. (Omitting it falls back to the most-recent window, which is only
    valid for a live pick, not a historical backtest.)"""
    d = _scorable(df, require_fwd)
    d["_c"] = neutralize(score(d, weights), d, sector_neutral, beta_neutral)
    order = list(d.sort_values("_c", ascending=False)["company_key"])
    M = D.daily_return_matrix()
    win = (M[M.index <= asof] if asof is not None else M).tail(lookback)
    chosen = []
    for ck in order:
        if ck not in win.columns:
            continue
        if not chosen:
            chosen.append(ck); continue
        pair = win[[ck] + chosen].corr().iloc[0, 1:].abs()
        c = pair.mean() if corr_mode == "mean" else pair.max()
        if pd.isna(c) or c <= cap:
            chosen.append(ck)
        if len(chosen) >= n:
            break
    if len(chosen) < n:                       # backfill if the cap was too tight
        for ck in order:
            if ck not in chosen:
                chosen.append(ck)
            if len(chosen) >= n:
                break
    return chosen[:n]


# ---- returns + evaluation ----------------------------------------------------
def daily_from_holdings(pan: EdgePanel, holds: list, hold: int = HOLD) -> pd.Series:
    M = D.daily_return_matrix(); cal = M.index
    pieces = []
    for i, d in enumerate(pan.bdates):
        cks = [c for c in holds[i] if c in M.columns]
        loc = int(cal.searchsorted(d, side="right"))
        win = cal[loc: loc + hold]
        if len(win) == 0 or not cks:
            continue
        pieces.append(M.loc[win, cks].mean(axis=1))
    if not pieces:
        return pd.Series(dtype=float)
    s = pd.concat(pieces)
    return s[~s.index.duplicated(keep="first")].fillna(0.0)


def period_returns(pan: EdgePanel, holds: list, regime_cash: bool = False,
                   weight: str = "equal") -> np.ndarray:
    """Top-N forward returns per period. If regime_cash, periods where the
    market is below its 200-day MA earn cash instead.

    `weight`:
      "equal"  -- 1/n each (the shipped book).
      "invvol" -- proportional to 1/vol_21, i.e. RISK parity rather than DOLLAR
                  parity. Equal dollars in a 20%-vol name and an 80%-vol name
                  is not a balanced book: the volatile name supplies most of
                  the portfolio's variance and dominates the outcome. Weights
                  use vol known AT the rebalance, so there is no look-ahead."""
    out = []
    for i, df in enumerate(pan.panels):
        sel = df[df["company_key"].isin(holds[i])]
        if not len(sel):
            out.append(pan.rf_per if regime_cash and not pan.ma200_on[i] else 0.0)
            continue
        fwd = pd.to_numeric(sel["fwd_ret"], errors="coerce")
        if weight == "invvol" and "vol_21" in sel.columns:
            v = pd.to_numeric(sel["vol_21"], errors="coerce")
            w = 1.0 / v.where(v > 0)
            w = w.fillna(w.median() if w.notna().any() else 1.0)
            r = float((fwd * w).sum() / w.sum()) if w.sum() > 0 else float(fwd.mean())
        else:
            r = float(fwd.mean())
        if regime_cash and not pan.ma200_on[i]:
            r = pan.rf_per
        out.append(r)
    return np.array(out)


def perf(rets, ppy=PPY):
    r = np.nan_to_num(np.asarray(rets, float))
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (ppy / len(r)) - 1 if len(r) and eq[-1] > 0 else -1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(r) else 0.0
    sh = float(np.sqrt(ppy) * np.nanmean(r) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
    return float(cagr), dd, sh


def sortino(rets, ppy=PPY):
    """Annualized return per unit of DOWNSIDE deviation (MAR=0) — like Sharpe but
    only losses count as risk. Same no-excess-of-cash convention as perf()/the page."""
    r = np.nan_to_num(np.asarray(rets, float))
    if len(r) == 0:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return float(np.sqrt(ppy) * np.mean(r) / dd) if dd > 0 else 0.0


def eval_windows(pan: EdgePanel, rets, label=""):
    """Print CAGR / Sharpe / maxDD over trailing 1Y / 2Y / 5Y / full, vs S&P,
    using the panel's true rebalances-per-year for annualization."""
    rets = np.asarray(rets, float)
    ppy = pan.ppy
    windows = (("1Y", int(round(ppy))), ("2Y", int(round(2 * ppy))),
               ("5Y", int(round(5 * ppy))), ("MAX", None))
    print(f"  {label}")
    print(f"    {'window':<6}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}   {'vs S&P CAGR':>12}")
    for wname, k in windows:
        kk = len(rets) if k is None else min(k, len(rets))
        c, dd, sh = perf(rets[-kk:], ppy)
        cs, _, _ = perf(pan.spxf[-kk:], ppy)
        print(f"    {wname:<6}{c*100:7.1f}%{sh:8.2f}{dd*100:7.0f}%   {(c-cs)*100:>+11.1f}%")
    return rets


def _bench_fwd_hold(series, d, hold):
    # Match the stock-side execution lag: enter at the first close after d (t+1)
    # and exit hold days later, so the benchmark is measured over the same window.
    fwd = series[series.index > d]
    if len(fwd) < hold + 1:
        return np.nan
    return float(fwd.iloc[hold] / fwd.iloc[0] - 1)


# ---- the robust, tradeable Edge: signal + liquidity + corr-cap + regime + costs
EDGE_SPEC = dict(
    # 2026-07-27: acceleration was RETIRED here. On survivorship-free Sharadar
    # data it ranked the cross-section at +0.27%/rebalance (t=0.4) and returned
    # -6.4%/yr against the S&P out of sample with a -52.8% drawdown. It is dead
    # in every size bucket, era, sector and regime tested. What replaced it:
    signal={"ret_12_1": 1.0},   # 12-1 momentum (Jegadeesh-Titman): 12-month
                                # return skipping the last month. The strongest
                                # signal in the scan, and CONDITIONAL -- it works
                                # in large caps (+3.15%/reb, t=2.6), in uptrends
                                # (+4.06%, t=2.7) and in high-dispersion sectors,
                                # which is why the floor and the gate below are
                                # part of the model rather than decoration.
    hold=42,                    # 2-month clock. vs 1-month: gives up 1.6pts of
                                # return for HALF the drawdown (-16% vs -33%)
                                # and the best Sharpe (0.97 vs 0.88).
    n=10,                       # the product book. Tighter is worse: n=5 draws
                                # down -57% and goes negative out of sample.
    mcap_floor=1e10,            # $10B. Momentum degrades monotonically as the
                                # floor drops -- at $200M the same model returns
                                # -14.5%/yr excess. This floor is load-bearing.
    fcf_screen=False,           # OFF. On the production daily basis the screen
                                # costs 3.8pts of CAGR and 0.06 Sharpe for no
                                # drawdown benefit (-46% vs -47%). It looked
                                # good only on the per-rebalance basis, which
                                # understated drawdown. Kept implemented for
                                # research; not shipped.
    corr_cap=None,              # OFF: it pushes the book away from the highest-
                                # momentum names, i.e. away from the signal.
                                # Costs 2.5pts of excess.
    corr_lookback=126, regime_expo=0.25, cost_bps=10.0,
    growth_mix=0.0,             # retired with accel: the revenue-growth gate
                                # ranked NEGATIVELY on honest data (-1.61%/reb,
                                # t=-2.5, negative in both halves).
    growth_thresh=0.15,
    sector_cap=2,               # max 2 names per sector, so a 10-name book
                                # spans >=5 sectors. The live book was 9/10
                                # Technology -- one industry shock owned the
                                # portfolio. Measured: -47% -> -41% drawdown
                                # for 2.6pts of CAGR, Sharpe 0.82 -> 0.80.
                                # Caps of 3/4/5 did NOT help (drawdown -48/-52/
                                # -49%) -- only a hard 2 forces real breadth.
    stagger=False,              # OFF: costs 0.8pts; adopted for accel.
    continuous_regime=True,     # ON as of 2026-07-29. The comment here used to
                                # say this cost 3pts of drawdown for 2.6pts of
                                # return -- that was measured on the OLD spec
                                # (accel signal, no sector cap, different floor)
                                # and no longer holds. Re-measured on the shipped
                                # spec it is better on EVERY axis: CAGR 19.48 ->
                                # 20.19%, Sharpe 0.801 -> 0.865, Sortino 1.135 ->
                                # 1.230, maxDD -41.29 -> -39.35%. It adds no new
                                # parameter (same 200dMA, judged daily instead of
                                # frozen at the rebalance) and wins in 3 of 4
                                # sub-eras. It fixes fast beta crashes -- COVID
                                # -41.3 -> -25.6%, 2015-16 -35.0 -> -21.4% -- and
                                # by construction cannot help a momentum unwind
                                # that happens with the market above its MA.
    # ---- volatility targeting (2026-07-29) --------------------------------
    # Scale exposure toward 25% annualized vol using the BOOK's own trailing
    # vol. This exists because the regime gate is structurally blind to the
    # model's two worst drawdowns: 2021-08 (-39.8%) and 2000-03 (-38.3%) both
    # happened with the S&P above its 200dMA 95-97% of the time while the index
    # fell only ~10%. Portfolio beta is 0.82, so those were momentum unwinds,
    # not market events -- and momentum vol spikes before momentum crashes.
    # Measured on top of continuous_regime: CAGR 20.19 -> 18.47%, Sharpe 0.865
    # (those figures predate the 2026-07-29 delisting-default change; the shipped
    # spec now reads 18.29% / 0.934 with dying names included -- see DELIST_HAIRCUT)
    # -> 0.94, maxDD -39.35 -> -30.8%. It is the only lever tested that improved
    # ALL FIVE drawdown episodes. Response is monotone in the target (25/20/15%
    # -> -30.8/-25.5/-21.0% DD), i.e. no magic value was fitted.
    # A name-level 200dMA filter was tested alongside and REJECTED: it fixed the
    # 2021 unwind (-39.8 -> -27.4%) but made the aggregate drawdown WORSE
    # (-44.3%) by shrinking the basket in bad tape.
    vol_target=0.25,            # None disables; cap below means de-lever only
    vol_lookback=21,            # trailing days for realized vol (causal, t-1)
    vol_cap=1.0)                # never above 100% invested -- no leverage

def _sector_capped(frame, k, cap):
    """Walk best-first, admitting a name only while its sector is under `cap`.

    A 10-name momentum book has no sector control at all: whatever sector is
    running takes the whole basket (the live book was 9/10 Technology). This
    is a RISK control justified before any backtest -- it bounds how much of
    the portfolio one industry shock can take -- not a fitted parameter.
    If the cap can't be filled (too few sectors represented), the remainder is
    back-filled in score order so the book is never short of names."""
    counts, chosen = {}, []
    order = list(frame["company_key"])
    secs = dict(zip(frame["company_key"], frame.get("sector", pd.Series(index=frame.index)).astype(str)))
    for ck in order:
        sec = secs.get(ck, "Unknown")
        if counts.get(sec, 0) >= cap:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        chosen.append(ck)
        if len(chosen) >= k:
            return chosen
    for ck in order:                      # back-fill rather than return short
        if ck not in chosen:
            chosen.append(ck)
        if len(chosen) >= k:
            break
    return chosen[:k]


def _blend_select(d, asof, n, weights, corr_cap, corr_lookback, growth_mix, growth_thresh,
                  require_fwd=True, corr_mode="max", sector_neutral=False,
                  beta_neutral=False, fcf_screen=False, sector_cap=None):
    """Build the basket as K growth-gated picks + (n-K) pure-signal picks, where
    K = round(growth_mix * n). growth_mix=0 -> pure signal; 1 -> all-growth.

    `require_fwd=False` picks from rows with no realized forward return -- only
    valid for the LIVE book (the still-open position), never for the backtest."""
    def pick(frame, k):
        if k <= 0 or len(frame) < 1:
            return []
        if corr_cap:
            return corr_cap_select(frame, asof=asof, n=k, weights=weights, cap=corr_cap,
                                   lookback=corr_lookback, require_fwd=require_fwd,
                                   corr_mode=corr_mode, sector_neutral=sector_neutral,
                                   beta_neutral=beta_neutral)
        f = _scorable(frame, require_fwd)
        f["_c"] = neutralize(score(f, weights), f, sector_neutral, beta_neutral)
        f = f.sort_values("_c", ascending=False)
        if sector_cap:
            return _sector_capped(f, k, int(sector_cap))
        return list(f.head(k)["company_key"])
    if fcf_screen and "fcf_margin" in d.columns:
        # Halve the pool on cash generation BEFORE ranking. Applied here so it
        # binds identically for the backtest and the live book.
        _m = pd.to_numeric(d["fcf_margin"], errors="coerce")
        _keep = (_m >= _m.median()).fillna(False)
        if _keep.sum() >= max(n * 2, 20):
            d = d[_keep]
    K = int(round(growth_mix * n))
    gpick = []
    if K > 0:
        g = d[pd.to_numeric(d["rev_growth"], errors="coerce") >= growth_thresh]
        gpick = pick(g, K)
    sel = list(gpick)
    for ck in pick(d, n + len(gpick)):          # fill remainder from the full pool
        if len(sel) >= n:
            break
        if ck not in sel:
            sel.append(ck)
    return sel[:n]


# The heavy layer. Its key space over the PUBLIC selector options is
# hold(4) x n(6) x growth_mix(5) = 120 distinct computations (windows are cheap
# slices of these, so they cost nothing extra). Size the cache ABOVE that space so
# entries are never evicted: total heavy work is then bounded at 120 per worker for
# the life of the process, no matter how many parameter combos get requested.
# Entries are small (a handful of ~250-float arrays), so this costs ~MBs.
def _candidates(df, mcap_floor, name_trend=False):
    """Per-rebalance candidate set: the size floor, plus optionally the name's own
    200dMA trend. Shared by _edge_full and _select_holds so the two selection
    loops cannot drift apart (they already duplicate the loop body)."""
    d = df[df["pit_mcap"] >= mcap_floor] if mcap_floor else df
    if name_trend and "px_ma200" in d.columns:
        d = d[d["px_ma200"] >= 1.0]
    return d


@lru_cache(maxsize=160)
def _edge_full(hold, n, mcap_floor, corr_cap, corr_lookback, regime_expo, cost_bps,
               signal_key, growth_mix=0.0, growth_thresh=0.15, fcf_screen=False,
               sector_cap=None, name_trend=False):
    """Compute the full-history Edge once (cached). Returns the per-period gross
    & net returns, turnover, and aligned S&P / Nasdaq returns.

    growth_mix in [0,1] sets how much of the basket must come from names with YoY
    revenue growth >= growth_thresh (a fundamentals tilt; 0 = pure market-data)."""
    weights = dict(signal_key)
    pan = load_edge_panel(hold=hold)
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = _candidates(df, mcap_floor, name_trend)
        cks = _blend_select(d, pan.bdates[i], n, weights, corr_cap, corr_lookback,
                            growth_mix, growth_thresh, fcf_screen=fcf_screen,
                            sector_cap=sector_cap)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    gross = period_returns(pan, holds)
    if regime_expo is not None:
        gross = np.where(pan.ma200_on, gross, regime_expo * gross + (1 - regime_expo) * pan.rf_per)
    turn = np.array(turn)
    net = gross - (cost_bps / 1e4) * turn
    # holds (per-rebalance company_key baskets) is returned too so callers like the
    # Edge Tracker can show what was actually held each rebalance, with no recompute.
    return pan.bdates, gross, net, turn, pan.spxf, pan.ndxf, holds


# ---- daily buy-and-hold simulation + staggered sleeves ----------------------
def _select_holds(pan, n, weights, mcap_floor, corr_cap, corr_lookback,
                  growth_mix, growth_thresh, fcf_screen=False, sector_cap=None,
                  name_trend=False):
    """Full-spec per-rebalance selection + membership turnover for one panel."""
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = _candidates(df, mcap_floor, name_trend)
        cks = _blend_select(d, pan.bdates[i], n, weights, corr_cap, corr_lookback,
                            growth_mix, growth_thresh, fcf_screen=fcf_screen,
                            sector_cap=sector_cap)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    return holds, np.array(turn)


@lru_cache(maxsize=64)
def _select_holds_cached(hold, offset_days, n, signal_key, mcap_floor, corr_cap,
                         corr_lookback, growth_mix, growth_thresh, fcf_screen=False,
                         sector_cap=None, name_trend=False):
    """Selection is the SAME for the net and gross backtest passes (cost doesn't
    change which names are picked), and it's the dominant cost (~8s/sleeve via the
    correlation cap). Cache it on the hashable spec so the gross pass — and repeat
    backtests at a different cost/window — reuse it instead of re-selecting."""
    pan = load_edge_panel(hold=hold, offset_days=offset_days)
    holds, turn = _select_holds(pan, n, dict(signal_key), mcap_floor, corr_cap,
                                corr_lookback, growth_mix, growth_thresh,
                                fcf_screen=fcf_screen, sector_cap=sector_cap,
                                name_trend=name_trend)
    return pan, holds, turn


@lru_cache(maxsize=1)
def _ma200_daily_state():
    """Daily boolean on the return-matrix calendar: is the S&P above its 200-day
    MA, judged at the PRIOR close (causal, no look-ahead)? True where the MA isn't
    yet defined (early history) -- matching the per-rebalance regime's convention
    of staying invested when the state is unknown."""
    M = D.daily_return_matrix(); cal = M.index
    spx = D.benchmarks()["SP500"].dropna()
    ma = spx.rolling(200).mean()
    on = pd.Series(np.where(ma.notna(), (spx > ma).to_numpy(), True), index=spx.index)
    on = on.reindex(cal, method="ffill").shift(1).fillna(True)        # prior-close state
    return on.to_numpy(bool)


def _sleeve_daily(pan, holds, turn, M, hold, regime_expo, cost_bps, regime_daily=None):
    """Daily return Series for one sleeve: buy-and-hold each book for `hold` days
    from t+1 (weights drift), regime scales the window, cost on the entry day.
    NaN outside the sleeve's active days so sleeves can be averaged cleanly.

    `regime_daily`: optional daily boolean array (S&P>200dMA, causal) aligned to
    M's calendar. When provided AND regime_expo is set, the de-risk is applied per
    DAY across the hold window (continuous regime) rather than frozen at the
    rebalance date. None -> the original per-rebalance behaviour."""
    cal = M.index
    out = pd.Series(np.nan, index=cal)
    rf_d = config.RISK_FREE_ANNUAL / 252.0
    for i, cks in enumerate(holds):
        loc = int(cal.searchsorted(pan.bdates[i], side="right"))     # t+1 entry (exec lag)
        win = cal[loc: loc + hold]
        names = [c for c in cks if c in M.columns]
        if len(win) == 0 or not names:
            continue
        sub = np.nan_to_num(M.loc[win, names].to_numpy(float))
        w = np.full(len(names), 1.0 / len(names))
        on_rebal = bool(pan.ma200_on[i]) if regime_expo is not None else True
        rets = np.empty(len(win))
        for dd in range(len(win)):
            r = sub[dd]; pr = float(np.dot(w, r))
            if regime_daily is not None and (loc + dd) < len(regime_daily):
                on = bool(regime_daily[loc + dd])        # continuous: this day's state
            else:
                on = on_rebal                            # per-rebalance (or fallback)
            if regime_expo is not None and not on:
                pr = regime_expo * pr + (1 - regime_expo) * rf_d
            rets[dd] = pr
            g = w * (1 + r); ssum = g.sum()
            if ssum > 0:
                w = g / ssum
        rets[0] -= (cost_bps / 1e4) * float(turn[i])                 # entry-day cost
        out.loc[win] = rets
    return out


def _apply_vol_target(model, target, lookback, cap, cost_bps):
    """Scale exposure toward a constant target volatility; the uninvested
    remainder earns the risk-free rate (the same convention the regime gate uses).

    WHY: the market regime gate cannot see a momentum unwind. The two worst
    drawdowns in this model's history (2021-08 -39.8%, 2000-03 -38.3%) happened
    with the S&P above its 200dMA 95-97% of the time while the index fell only
    ~10% -- portfolio beta is 0.82, so those were factor events, not market
    events. The book's OWN volatility does see them: it spikes before momentum
    crashes. This is the only lever tested that touched all five episodes.

    CAUSAL: the scale applied on day t comes from returns through t-1 (.shift(1)).
    Using day t's own volatility would be look-ahead -- de-risking on days it
    already knows are bad, which makes any vol target look brilliant.

    Re-levering is a trade, so the change in exposure is charged turnover at the
    same rate as a rebalance. Omitting that would flatter a lever whose entire
    mechanism is trading more often.
    """
    rv = model.rolling(lookback).std().shift(1) * np.sqrt(252.0)
    scale = (target / rv).clip(upper=cap).fillna(cap)
    rf_d = config.RISK_FREE_ANNUAL / 252.0
    scaled = scale * model + (1.0 - scale) * rf_d
    relever = scale.diff().abs().fillna(0.0)
    return scaled - (cost_bps / 1e4) * relever


@lru_cache(maxsize=16)
def _edge_daily(hold, n, mcap_floor, corr_cap, corr_lookback, regime_expo, cost_bps,
                signal_key, growth_mix, growth_thresh, stagger, continuous_regime=False,
                fcf_screen=False, sector_cap=None, name_trend=False,
                vol_target=None, vol_lookback=21, vol_cap=1.0):
    """Daily NET return series for the tradeable Edge (staggered sleeves when
    stagger=True). Returns aligned daily model / S&P / Nasdaq returns + turnover +
    the primary sleeve's holds. maxDD taken on THIS daily curve = true peak-to-trough.

    continuous_regime=True evaluates the 200dMA de-risk per DAY across each hold
    window (a robust drawdown reducer) instead of freezing it at the rebalance.

    vol_target (e.g. 0.25) scales daily exposure toward that annualized vol; see
    _apply_vol_target. vol_cap=1.0 means de-lever only -- never borrow."""
    weights = dict(signal_key)
    M = D.daily_return_matrix()
    rd = _ma200_daily_state() if (continuous_regime and regime_expo is not None) else None
    offsets = (0, hold // 2) if stagger else (0,)
    series, turns, prim_holds = [], [], None
    for off in offsets:
        pan, holds, turn = _select_holds_cached(hold, off, n, signal_key, mcap_floor,
                                                corr_cap, corr_lookback, growth_mix,
                                                growth_thresh, fcf_screen, sector_cap,
                                                name_trend)
        series.append(_sleeve_daily(pan, holds, turn, M, hold, regime_expo, cost_bps, rd))
        turns.append(float(np.mean(turn)))
        if off == 0:
            prim_holds = (pan.bdates, holds)
    both = pd.concat(series, axis=1)
    model = both.mean(axis=1, skipna=True).dropna()      # 50/50 where both active
    if vol_target:
        model = _apply_vol_target(model, vol_target, vol_lookback, vol_cap, cost_bps)
    idx = model.index
    bm = D.benchmarks()
    spx = bm["SP500"].reindex(idx, method="ffill").pct_change().fillna(0.0)
    ndx = bm["NASDAQ"].reindex(idx, method="ffill").pct_change().fillna(0.0)
    return (idx, model.to_numpy(), spx.to_numpy(), ndx.to_numpy(),
            float(np.mean(turns)), prim_holds)


def _perf_daily(r):
    """CAGR / maxDD / Sharpe from a DAILY net-return array (annualized at 252)."""
    r = np.nan_to_num(np.asarray(r, float))
    if len(r) < 2:
        return 0.0, 0.0, 0.0
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (252.0 / len(r)) - 1 if eq[-1] > 0 else -1.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sh = float(np.sqrt(252) * np.nanmean(r) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
    return float(cagr), dd, sh


def _sortino_daily(r):
    r = np.nan_to_num(np.asarray(r, float))
    dn = np.minimum(r, 0.0); d = float(np.sqrt(np.mean(dn ** 2)))
    return float(np.sqrt(252) * np.mean(r) / d) if d > 0 else 0.0


def reset_caches() -> None:
    """Drop every memoized result that depends on the panel.

    REQUIRED after mutating the module-level research knobs USE_PIT_UNIVERSE or
    DELIST_HAIRCUT: neither is part of any cache key, so a surviving entry
    silently returns the PREVIOUS variant's numbers -- a wrong research result
    with no error. Clearing load_edge_panel alone is not enough; the selection
    and daily-series caches sit above it and would still be warm.
    (_ma200_daily_state is not cleared: it depends only on the S&P calendar.)

    The on-disk panel cache needs no clearing: USE_PIT_UNIVERSE and
    DELIST_HAIRCUT are hashed into its fingerprint, so a knob flip misses and
    rebuilds instead of returning the other variant's panel."""
    for fn in (load_edge_panel, _select_holds_cached, _edge_full, _edge_daily):
        fn.cache_clear()


_SIGNAL_LABELS = {
    "ret_12_1": "12-1 momentum (12-month return, skipping the last month)",
    "ret_126": "6-month momentum", "ret_63": "3-month momentum",
    "ret_21": "1-month reversal", "accel": "acceleration (3m − prior 3m)",
    "hi_252": "52-week-high proximity", "rs_63": "relative strength vs the market",
}


def _signal_label(signal_key) -> str:
    """Human label for whatever is ACTUALLY being traded. This was hardcoded to
    "acceleration" and kept saying so after the signal changed -- the page would
    have described a model that is no longer running."""
    parts = dict(signal_key) if not isinstance(signal_key, dict) else signal_key
    return " + ".join(_SIGNAL_LABELS.get(k, k) for k in parts) or "unknown"


def run_edge_backtest(window: str = "MAX", spec: dict | None = None) -> dict:
    """Backtest the tradeable Edge over a trailing window. Returns curves +
    performance in the same shape the website's chart code expects (NET of costs)."""
    s = {**EDGE_SPEC, **(spec or {})}
    hold = s["hold"]
    stagger = bool(s.get("stagger", True))
    sig = tuple(sorted(s["signal"].items()))
    nt = s.get("name_trend", False)
    vt = s.get("vol_target"); vlb = s.get("vol_lookback", 21); vcap = s.get("vol_cap", 1.0)
    gm = float(s.get("growth_mix", 0.0) or 0.0); gt = float(s.get("growth_thresh", 0.15))
    cr = bool(s.get("continuous_regime", False))
    fs = bool(s.get("fcf_screen", False))
    sc_cap = s.get("sector_cap")
    idx, model, spx, ndx, avg_to, prim = _edge_daily(
        hold, s["n"], s["mcap_floor"], s["corr_cap"], s["corr_lookback"],
        s["regime_expo"], s["cost_bps"], sig, gm, gt, stagger, cr, fs, sc_cap, nt,
        vt, vlb, vcap)
    # gross (cost-free) daily model, for the gross->net turnover card
    gidx, gmodel, *_ = _edge_daily(hold, s["n"], s["mcap_floor"], s["corr_cap"],
                                   s["corr_lookback"], s["regime_expo"], 0.0, sig, gm, gt,
                                   stagger, cr, fs, sc_cap, nt, vt, vlb, vcap)
    T = len(model)
    k = window_k(window, T)
    if k < 20:
        return {"ok": False, "reason": "window too short"}
    sl = slice(T - k, T)
    mr, sr, nr = model[sl], spx[sl], ndx[sl]
    dts = idx[T - k:T]

    def curve(r):                                    # growth-of-$1, subsampled for payload
        eq = np.cumprod(1 + np.nan_to_num(r))
        stride = max(1, len(eq) // 300)
        keep = list(range(0, len(eq), stride))
        if keep[-1] != len(eq) - 1:
            keep.append(len(eq) - 1)
        return keep, [round(float(eq[i]), 4) for i in keep]

    def stats(r):
        c, dd, sh = _perf_daily(r)
        tot = float(np.cumprod(1 + np.nan_to_num(r))[-1] - 1)
        return {"cagr": c, "sharpe": sh, "sortino": _sortino_daily(r),
                "max_drawdown": dd, "total_return": tot}

    windows = []
    for wn in SUMMARY_WINDOWS:
        kk = window_k(wn, T)
        cg, dd, sh = _perf_daily(model[T - kk:T]); cs, _, _ = _perf_daily(spx[T - kk:T])
        windows.append({"w": wn, "cagr": cg, "sharpe": sh, "dd": dd,
                        "sp_cagr": cs, "excess": cg - cs})
    gc = _perf_daily(gmodel[len(gmodel) - min(k, len(gmodel)):])[0]

    # best / worst single hold-length window (rolling), from the daily net series
    mser = pd.Series(mr, index=dts)
    roll = (1 + mser).rolling(hold).apply(np.prod, raw=True) - 1
    roll = roll.dropna()
    bi = roll.idxmax(); wi = roll.idxmin()
    period_detail = {
        "hold_days": hold,
        "best": {"ret": float(roll.max()), "date": str(bi.date())},
        "worst": {"ret": float(roll.min()), "date": str(wi.date())},
        "pretax_total": stats(mr)["total_return"],   # net of costs, before taxes
        "pretax_cagr": stats(mr)["cagr"],
    }
    keep, mcurve = curve(mr)
    dstr = [str(dts[i].date()) for i in keep]
    ann_to = round(avg_to * (252.0 / hold), 1)
    return {
        "ok": True, "n_rebalances": int(k / hold),
        "period": [str(dts[0].date()), str(dts[-1].date())],
        "period_detail": period_detail,
        "spec": {"signal": _signal_label(sig), "hold_days": hold,
                 "n": s["n"], "mcap_floor_bn": s["mcap_floor"] / 1e9,
                 "corr_cap": s["corr_cap"], "regime_expo": s["regime_expo"],
                 "cost_bps": s["cost_bps"], "stagger": stagger,
                 "growth_mix": gm, "growth_thresh": gt, "continuous_regime": cr,
                 # surfaced so the page describes what actually runs (rules #7)
                 "sector_cap": sc_cap, "fcf_screen": fs},
        "performance": {"model": stats(mr), "sp500": stats(sr), "nasdaq": stats(nr)},
        "curves": {"dates": dstr, "model": mcurve,
                   "sp500": curve(sr)[1], "nasdaq": curve(nr)[1]},
        "windows": windows,
        "turnover": {"per_rebalance": round(avg_to, 3), "annualized": ann_to,
                     "cost_bps": s["cost_bps"], "gross_cagr": gc,
                     "net_cagr": stats(mr)["cagr"]},
    }


if __name__ == "__main__":
    print("Building Edge trading panel (one-time, ~1-2 min) ...")
    pan = load_edge_panel()
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}, "
          f"avg {np.mean([len(p) for p in pan.panels]):.0f} names; "
          f"market above 200dMA {pan.ma200_on.mean()*100:.0f}% of the time\n")
    holds = [select_topN(df) for df in pan.panels]
    base = period_returns(pan, holds)
    eval_windows(pan, base, "BASELINE: 3-month momentum, equal-weight top-10, no overlays")
