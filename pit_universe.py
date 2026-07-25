"""Pluggable point-in-time (PIT) Russell-1000 universe layer for the Edge.

WHY THIS EXISTS
---------------
The Edge currently backtests on a Russell-1000 *proxy*: at each rebalance it
takes the top ~1000 names by point-in-time market cap from the cached fiscal.ai
data. That cache holds 1111 companies of which only 111 are flagged "Inactive"
(delisted) -- i.e. ~89% of the names that ever left the index are missing. The
result is survivorship-inflated history (commonly +3-5%/yr of fake CAGR).

The #1 credibility fix is REAL PIT Russell-1000 constituents + delisted returns.
This module is the data layer for that fix. It exposes a single function the
Edge calls --

    pit_members(asof_date) -> set[company_key]

-- backed by a chain of adapters tried in priority order:

    (a) NorgateAdapter   -- real PIT membership + delisted securities + adjusted
                            returns, if the `norgatedata` SDK and a Norgate
                            database are present (the recommended retail source).
    (b) CSVAdapter       -- a user-dropped membership file at
                            data/russell1000_membership.csv (cols: date,ticker)
                            plus an optional delisted-returns file.
    (c) ProxyAdapter     -- the current fiscal.ai top-1000-by-PIT-mcap behaviour,
                            so nothing ever breaks. (NOT real PIT -- still
                            survivorship biased; clearly labelled as such.)

HONESTY
-------
This module NEVER fabricates membership or returns. If no real source is present
the active adapter is the proxy and `is_real_pit()` returns False -- callers (and
the user) are told plainly that the de-biasing is not yet in effect.

Tickers are mapped to the Edge's company_key (format "EXCHANGE_TICKER", e.g.
"NASDAQ_NVDA") using the meta in the cached backtest data.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

# ---------------------------------------------------------------------------
# paths / config
# ---------------------------------------------------------------------------
DATA_DIR = Path(config.__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "cache"
# Default path auto-activates the CSV adapter (and thus the PIT universe) when
# present -- reserve it for REAL Russell-1000 membership. Research runs against
# other membership files (e.g. data/sp500_membership.csv, fetched by
# fetch_membership.py) should point PIT_MEMBERSHIP_CSV at them explicitly so
# the product's proxy behaviour never flips silently.
MEMBERSHIP_CSV = Path(os.environ.get("PIT_MEMBERSHIP_CSV",
                                     DATA_DIR / "russell1000_membership.csv"))
DELISTED_RETURNS = DATA_DIR / "delisted_returns.parquet"   # or .csv
DELISTED_RETURNS_CSV = DATA_DIR / "delisted_returns.csv"
MEMBERSHIP_ARTIFACT = CACHE_DIR / "pit_membership.json"     # cached built artifact

NORGATE_WATCHLIST = os.environ.get("NORGATE_RUSSELL1000_WATCHLIST", "Russell 1000 Current & Past")

# Force a particular adapter for testing: NORGATE_ONLY / CSV_ONLY / PROXY_ONLY.
FORCE_ADAPTER = os.environ.get("PIT_FORCE_ADAPTER", "").strip().upper()


# ---------------------------------------------------------------------------
# ticker <-> company_key mapping (from the cached fiscal.ai meta)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ticker_to_key() -> dict:
    """Map plain ticker -> company_key using the cached backtest meta.

    company_key looks like 'NASDAQ_NVDA'; meta carries the bare ticker 'NVDA'.
    We also key on a normalised ticker (dots/dashes stripped) so 'BRK.A' and
    'BRK-A' both resolve.
    """
    import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)
    data = engine.load_bt_data()
    out: dict[str, str] = {}
    for ck, blob in data.items():
        meta = blob.get("meta") or {}
        tk = meta.get("ticker")
        if not tk:
            # fall back to the part after the first underscore in the key
            tk = ck.split("_", 1)[-1]
        out.setdefault(str(tk).upper(), ck)
        out.setdefault(_norm_ticker(tk), ck)
    return out


def _norm_ticker(t: str) -> str:
    return str(t).upper().replace(".", "").replace("-", "").replace(" ", "")


def tickers_to_keys(tickers) -> set:
    """Resolve an iterable of tickers to company_keys present in the cache.

    Tickers with no match in the cache are dropped (we cannot price them), and
    the count of unmatched names is recorded for reporting via `last_match_stats`.
    """
    m = _ticker_to_key()
    keys, matched, unmatched = set(), 0, []
    for t in tickers:
        ck = m.get(str(t).upper()) or m.get(_norm_ticker(t))
        if ck:
            keys.add(ck); matched += 1
        else:
            unmatched.append(t)
    _MATCH_STATS["matched"] = matched
    _MATCH_STATS["unmatched"] = len(unmatched)
    _MATCH_STATS["unmatched_sample"] = unmatched[:25]
    return keys


_MATCH_STATS: dict = {"matched": 0, "unmatched": 0, "unmatched_sample": []}


def last_match_stats() -> dict:
    return dict(_MATCH_STATS)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
class BaseAdapter:
    name = "base"
    is_real = False

    def available(self) -> bool:
        raise NotImplementedError

    def members(self, asof: pd.Timestamp) -> set:
        """Return the set of company_keys that were Russell-1000 members on
        `asof`. Empty set means 'this adapter cannot answer'."""
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {"adapter": self.name, "real_pit": self.is_real}


class NorgateAdapter(BaseAdapter):
    """Real PIT membership via the Norgate Data SDK.

    Norgate distributes a 'Russell 1000 Current & Past' index watchlist plus
    delisted securities with split/dividend-adjusted prices. We query index
    constituency at `asof` using norgatedata.index_constituent_timeseries (or
    the watchlist membership timeseries), then map the returned symbols to the
    Edge's company_keys.

    Requires: the Norgate Data Updater desktop app running with an active
    subscription, and `pip install norgatedata`.
    """
    name = "norgate"
    is_real = True

    def __init__(self):
        self._nd = None
        self._symbols = None

    def _load(self):
        if self._nd is not None:
            return self._nd
        import norgatedata  # raises ImportError if not installed
        self._nd = norgatedata
        return self._nd

    def available(self) -> bool:
        try:
            nd = self._load()
        except Exception:
            return False
        # database reachable? a status call will raise if the Updater/db is down.
        try:
            status = nd.status()
            return bool(status)
        except Exception:
            try:
                # alternative liveness check: can we enumerate the watchlist?
                syms = nd.watchlist_symbols(NORGATE_WATCHLIST)
                return bool(syms)
            except Exception:
                return False

    def members(self, asof: pd.Timestamp) -> set:
        nd = self._load()
        asof_d = pd.Timestamp(asof).normalize()
        if self._symbols is None:
            self._symbols = nd.watchlist_symbols(NORGATE_WATCHLIST)
        live_tickers = []
        for sym in self._symbols:
            try:
                ts = nd.index_constituent_timeseries(
                    sym, "Russell 1000",
                    padding_setting=nd.PaddingType.NONE,
                    timeseriesformat="pandas-dataframe",
                )
            except Exception:
                continue
            if ts is None or ts.empty:
                continue
            sl = ts[ts.index <= asof_d]
            if sl.empty:
                continue
            # column is typically 'Index Constituent' -> 1.0 when in the index
            val = sl.iloc[-1]
            flag = float(val.iloc[0] if hasattr(val, "iloc") else val)
            if flag >= 0.5:
                live_tickers.append(_strip_norgate(sym))
        return tickers_to_keys(live_tickers)

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        try:
            d["watchlist"] = NORGATE_WATCHLIST
            d["n_symbols"] = len(self._symbols) if self._symbols else None
        except Exception:
            pass
        return d


def _strip_norgate(sym: str) -> str:
    """Norgate symbols may carry a market suffix or delisting marker; keep the
    leading alnum/dot token (e.g. 'AAPL-202401' -> 'AAPL')."""
    s = str(sym).strip().upper()
    for sep in ("-", " ", "."):
        # keep dotted class shares (BRK.A) intact -- only split on the delisting
        # date marker pattern, which Norgate writes as a trailing '-<digits>'.
        pass
    # Norgate delisted symbols look like "TICKER-YYYYMMDD" or "TICKER-1".
    base = s.split("-")[0]
    return base


class CSVAdapter(BaseAdapter):
    """Real PIT membership from a user-dropped CSV.

    Expected file: data/russell1000_membership.csv with columns (case-insensitive):
        date,ticker
    one row per (rebalance/effective date, member ticker). Membership on an
    arbitrary `asof` is taken as the snapshot on the most recent date <= asof.

    Optional: data/delisted_returns.csv|.parquet with columns date,ticker,ret
    (daily total return) -- used to extend the priced universe with names the
    fiscal.ai cache is missing. (Wired through `delisted_returns()`.)
    """
    name = "csv"
    is_real = True

    def __init__(self):
        self._df = None

    def available(self) -> bool:
        return MEMBERSHIP_CSV.exists()

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        df = pd.read_csv(MEMBERSHIP_CSV)
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date"); tcol = cols.get("ticker")
        if dcol is None or tcol is None:
            raise ValueError(
                f"{MEMBERSHIP_CSV.name} must have 'date' and 'ticker' columns; "
                f"got {list(df.columns)}"
            )
        df = df[[dcol, tcol]].rename(columns={dcol: "date", tcol: "ticker"})
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df.dropna().sort_values("date")
        self._df = df
        return df

    @property
    def _snapshot_dates(self):
        return self._load()["date"].drop_duplicates().sort_values()

    def members(self, asof: pd.Timestamp) -> set:
        df = self._load()
        asof = pd.Timestamp(asof)
        dates = df["date"].drop_duplicates().sort_values()
        prior = dates[dates <= asof]
        if prior.empty:
            # asof is before the first snapshot -- use the earliest snapshot so the
            # backtest can still start (better than an empty universe).
            snap_date = dates.iloc[0]
        else:
            snap_date = prior.iloc[-1]
        tk = df.loc[df["date"] == snap_date, "ticker"].tolist()
        return tickers_to_keys(tk)

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        try:
            dts = self._snapshot_dates
            d["n_snapshots"] = int(len(dts))
            d["date_range"] = [str(dts.iloc[0].date()), str(dts.iloc[-1].date())]
        except Exception as e:
            d["error"] = str(e)
        return d


class ProxyAdapter(BaseAdapter):
    """Fallback: NOT real PIT. Returns an empty set so the Edge knows to use its
    existing top-1000-by-PIT-mcap logic. Exists only so the chain always resolves
    and the Edge never breaks when no real source is present."""
    name = "proxy"
    is_real = False

    def available(self) -> bool:
        return True

    def members(self, asof: pd.Timestamp) -> set:
        return set()  # sentinel: "use the legacy proxy in load_edge_panel"


# ---------------------------------------------------------------------------
# adapter resolution
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def active_adapter() -> BaseAdapter:
    """Pick the first available adapter in priority order (Norgate > CSV > proxy),
    honouring PIT_FORCE_ADAPTER if set."""
    candidates = [NorgateAdapter(), CSVAdapter(), ProxyAdapter()]
    if FORCE_ADAPTER:
        wanted = {"NORGATE": "norgate", "CSV": "csv", "PROXY": "proxy"}.get(FORCE_ADAPTER)
        candidates = [c for c in candidates if c.name == wanted] or [ProxyAdapter()]
    for c in candidates:
        try:
            if c.available():
                return c
        except Exception:
            continue
    return ProxyAdapter()


def is_real_pit() -> bool:
    """True iff a real PIT membership source is active (Norgate or CSV)."""
    return active_adapter().is_real


@lru_cache(maxsize=4096)
def pit_members(asof_date) -> frozenset:
    """Russell-1000 member company_keys as of `asof_date`.

    Returns an EMPTY set when the proxy adapter is active (the Edge then falls
    back to its top-1000-by-mcap behaviour). Cached per date.
    """
    asof = pd.Timestamp(asof_date)
    return frozenset(active_adapter().members(asof))


# ---------------------------------------------------------------------------
# delisted returns (CSV/Norgate) -- optional enrichment of the priced universe
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def delisted_returns() -> pd.DataFrame:
    """Daily returns for delisted names the fiscal.ai cache lacks.

    Columns: date,ticker,ret (daily total return as a decimal). Empty frame if
    no file is present. The Edge can splice these into its return matrix; today
    the cache already carries 111 inactive names so this is supplementary.
    """
    path = None
    if DELISTED_RETURNS.exists():
        path = DELISTED_RETURNS
    elif DELISTED_RETURNS_CSV.exists():
        path = DELISTED_RETURNS_CSV
    if path is None:
        return pd.DataFrame(columns=["date", "ticker", "ret"])
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols.get("date", "date"): "date",
                            cols.get("ticker", "ticker"): "ticker",
                            cols.get("ret", "ret"): "ret"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df[["date", "ticker", "ret"]]


# ---------------------------------------------------------------------------
# membership artifact builder (caches the resolved snapshots to data/cache/)
# ---------------------------------------------------------------------------
def build_membership_artifact(asof_dates) -> dict:
    """Resolve membership for each date in `asof_dates`, write a JSON artifact to
    data/cache/pit_membership.json, and return a summary. No-op-safe under the
    proxy adapter (records that real PIT is not active)."""
    ad = active_adapter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for d in asof_dates:
        ts = pd.Timestamp(d)
        keys = sorted(ad.members(ts))
        snapshots[str(ts.date())] = keys
    artifact = {
        "adapter": ad.name,
        "real_pit": ad.is_real,
        "n_dates": len(snapshots),
        "diagnostics": ad.diagnostics(),
        "match_stats": last_match_stats(),
        "snapshots": snapshots,
    }
    MEMBERSHIP_ARTIFACT.write_text(json.dumps(artifact, indent=0), encoding="utf-8")
    return {k: v for k, v in artifact.items() if k != "snapshots"}


def status() -> dict:
    """One-call summary of what the PIT layer can actually do right now."""
    ad = active_adapter()
    return {
        "active_adapter": ad.name,
        "real_pit": ad.is_real,
        "norgate_importable": _can_import("norgatedata"),
        "membership_csv_present": MEMBERSHIP_CSV.exists(),
        "delisted_returns_present": DELISTED_RETURNS.exists() or DELISTED_RETURNS_CSV.exists(),
        "diagnostics": ad.diagnostics(),
    }


def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import pprint
    pprint.pprint(status())
