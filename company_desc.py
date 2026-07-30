"""Per-holding write-ups for the Vision book. No paid API, nothing invented.

WHAT EACH HOLDING GETS
    industry  -- Sharadar's own label ("Semiconductors"). Metadata, not prose.
    what      -- what the business does, one or two sentences.
    standing  -- where the name sits in the field it was picked from.
    why       -- why THIS MODEL bought it.

WHY THERE IS NO LANGUAGE MODEL HERE
An earlier version generated all three sentences with Claude. That is off the
table: the API account has no funding, and a feature that silently stops working
when a balance runs out is not a pipeline. Everything below either comes from a
source we already pull or is COMPUTED from the model's own numbers.

That turned out to be the better design anyway. "Why we bought it" is the
sentence most likely to become fiction -- a language model asked why a fund owns
a stock will happily produce a moat-and-TAM thesis. The Edge has no view on
moats. It buys 12-1 momentum above a size floor under a sector cap, and that
reason is a handful of numbers we already have. Writing it as a template means it
cannot drift from what the model actually did, and it costs nothing to run.

WHERE `what` COMES FROM
yfinance -- the same Yahoo source this app already uses for its benchmark series
(edge_data.benchmarks), so it is not a new dependency or a new relationship. The
text is Yahoo's, trimmed to the first sentence or two, and cached permanently per
ticker because a company's line of business does not change month to month.

If you would rather not republish vendor text on a public page, set
EDGE_DESC_SOURCE=metadata and `what` is built from Sharadar's industry, scale and
location instead -- entirely our own data, at the cost of a thinner sentence.

FAILURE BEHAVIOUR
Never raises, never blocks a publish. Yahoo unreachable or rate-limited means the
`what` line falls back to the metadata sentence; everything else is local
arithmetic and always works. A description problem must not be able to stop the
book going out, and must never change a number.
"""
from __future__ import annotations
import json
import logging
import os
import pathlib
import re
import threading

import config

log = logging.getLogger(__name__)

CACHE_PATH = pathlib.Path(os.environ.get(
    "EDGE_DESC_PATH", str(config.CACHE_DIR / "company_desc.json")))
# Repo copy first, bundle copy second. The repo copy deploys with the code; the
# artifact-dir path is kept so an older bundle that carries one still works.
_META_CANDIDATES = (pathlib.Path(__file__).parent / "company_meta.json",
                    config.ARTIFACT_DIR / "company_meta.json")
META_PATH = next((p for p in _META_CANDIDATES if p.exists()), _META_CANDIDATES[0])
# "yahoo" (default) or "metadata" -- see the module docstring.
SOURCE = os.environ.get("EDGE_DESC_SOURCE", "yahoo").strip().lower()

# One writer at a time: gunicorn runs --threads 4 and two clicks of "Sync to
# Vision" can land together, and a read-modify-write of one JSON file from two
# threads loses one thread's work. Held across the fetch too, so the second click
# waits and then finds the first click's results already cached rather than
# hitting Yahoo again for the same tickers.
_LOCK = threading.Lock()

_SCALE = {"1 - Nano": "nano-cap", "2 - Micro": "micro-cap", "3 - Small": "small-cap",
          "4 - Mid": "mid-cap", "5 - Large": "large-cap", "6 - Mega": "mega-cap"}


# ---- cache -----------------------------------------------------------------
def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt cache must never break an export -- refetching always works.
        log.warning("description cache %s unreadable; starting fresh", CACHE_PATH,
                    exc_info=True)
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(tmp, CACHE_PATH)      # atomic; a reader never sees half a file
    except Exception:
        log.warning("could not write description cache %s", CACHE_PATH, exc_info=True)


def _company_meta() -> dict:
    """Sharadar industry/location sidecar (see build_company_meta.py)."""
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        log.warning("company_meta.json unreadable", exc_info=True)
        return {}


def industry_for(ticker: str) -> str:
    m = _company_meta().get(str(ticker).upper(), {})
    return m.get("industry") or m.get("sicindustry") or ""


# ---- the four fields -------------------------------------------------------
def _trim(summary: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    """First sentence or two of a business summary.

    Yahoo's summaries run to a wall of text ("...in the United States, Europe,
    the Middle East, Africa, Asia, and internationally") and the ask was for a
    MINI description. Split on sentence ends only where the next character is a
    space and a capital, so "Inc." and "U.S.A." don't cut the sentence in half.
    """
    s = " ".join(str(summary or "").split())
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", s)
    out = " ".join(parts[:max_sentences]).strip()
    if len(out) > max_chars:
        out = " ".join(parts[:1]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


def _metadata_sentence(ticker: str, name: str, m: dict) -> str:
    """`what` built only from data we license. Thinner than Yahoo's prose, but
    ours, and it always works."""
    industry = m.get("industry") or m.get("sicindustry") or ""
    scale = _SCALE.get(m.get("scalemarketcap", ""), "")
    loc = m.get("location") or ""
    bits = [b for b in (scale, industry.lower() if industry else "") if b]
    lead = f"{name} is a " + " ".join(bits) + " company" if bits else f"{name} is listed"
    if loc:
        lead += f", based in {loc}"
    return lead + "."


def _yahoo_summary(ticker: str) -> str:
    """Yahoo's business summary. Returns '' on any failure -- offline, rate
    limited, delisted, schema change. The caller falls back to metadata."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return _trim(info.get("longBusinessSummary") or "")
    except Exception as e:                                      # noqa: BLE001
        log.info("yahoo summary unavailable for %s: %s: %s", ticker,
                 type(e).__name__, e)
        return ""


def _fmt_move(v) -> str:
    """A trailing return, written the way a person would say it.

    Momentum books collect extreme winners, and this model's universe really does
    produce them: SNDK's 12-1 reading in the 2026-07 book was +3818%, verified
    against two independent sources (Sharadar's adjusted close and Yahoo's
    price x shares both land near a $180B cap on a ~$1,260 share price -- it
    genuinely rose ~30x). Printing "+3817.8%" is accurate and reads like a
    formatting bug, so past +500% this switches to a multiple."""
    f = float(v)
    if f >= 5.0:
        return f"a {f + 1:.0f}x rise"
    return f"a {f * 100:+.1f}% move"


def _standing(h: dict) -> str:
    """Where this name sits in the field it was chosen from. Pure arithmetic on
    numbers the tracker already computed (see _current_book).

    NOTE the field is the POST-SCREEN universe: since 2026-07-30 it counts only
    names that already cleared the size floor AND both solvency screens. Saying
    "top 1% of the companies that cleared the model's filters" would be wrong if
    it were measured before the filters ran."""
    pct, n = h.get("pctile"), h.get("n_eligible")
    if pct is None or not n:
        return ""
    top = max(0.1, round((1.0 - float(pct)) * 100, 1))
    mc = h.get("mcap")
    size = ""
    if mc:
        bn = f"{float(mc)/1e9:,.0f}"
        art = "an" if bn[0] in "8" or bn.startswith("11") or bn.startswith("18") else "a"
        size = f" It carried {art} ${bn}B market cap at the rebalance."
    return (f"Its momentum reading put it in the top {top:g}% of the {int(n):,} "
            f"companies that cleared the model's size and solvency filters this "
            f"month.{size}")


def _why(h: dict, rank: int, cfg: dict) -> str:
    """The model's actual reason, assembled from the actual numbers.

    Deliberately mechanical. This model does not read a filing or hold a view on
    any company; saying anything richer here would be inventing a thesis it does
    not have."""
    sig = h.get("signal")
    sig_txt = (f"{_fmt_move(sig)} over the last twelve months, excluding the most "
               f"recent month — the only thing this model ranks on"
               if isinstance(sig, (int, float)) and sig == sig else
               "its 12-1 momentum reading, the only thing this model ranks on")
    why = f"Ranked {rank} of {cfg.get('n', 10)} in this month's book on {sig_txt}."

    # The solvency screens are part of WHY a name is here -- momentum alone no
    # longer gets a stock into this book. Stated with the company's own figure
    # where we have it, so the sentence is checkable rather than boilerplate.
    gates = []
    if cfg.get("fcf_positive"):
        fcfm = h.get("fcf_margin")
        gates.append(f"generates positive free cash flow"
                     + (f" (margin {float(fcfm) * 100:.0f}%)"
                        if isinstance(fcfm, (int, float)) and fcfm == fcfm else ""))
    dmax = cfg.get("debt_ebitda_max")
    if dmax:
        de = h.get("debt_ebitda")
        gates.append(f"carries debt under {dmax:g}x EBITDA"
                     + (f" (at {float(de):.1f}x)"
                        if isinstance(de, (int, float)) and de == de else ""))
    if gates:
        why += (" It also passed the model's solvency screens: it "
                + " and ".join(gates) + ".")

    cap = cfg.get("sector_cap")
    sec = h.get("sector")
    if cap and sec:
        why += (f" It holds one of the {cap} places the model allows any single "
                f"sector, which is why a stronger {sec} name may be absent.")
    return why


# ---- entry point -----------------------------------------------------------
def describe(book: list[dict], book_date: str, cfg: dict | None = None) -> dict:
    """{ticker: {industry, what, standing, why}} for every holding in `book`.

    `book` rows come from edge_tracker_lib._current_book, so they carry signal,
    sector, pctile, n_eligible and mcap. `cfg` is the backtest spec dict.

    Always returns an entry per holding. Only `what` can come up empty, and only
    if Yahoo is unreachable AND the metadata sidecar is missing.
    """
    cfg = cfg or {}
    meta = _company_meta()
    out: dict[str, dict] = {}

    with _LOCK:
        cache = _load_cache()
        dirty = False
        for i, h in enumerate(book):
            t = str(h.get("ticker") or "").upper()
            if not t:
                continue
            m = meta.get(t, {})
            name = h.get("name") or m.get("name") or t

            # `what` is cached by TICKER ALONE, with no book date in the key: a
            # company's line of business does not change between rebalances, so
            # re-fetching it monthly would be ten needless network calls for
            # identical text. `standing` and `why` are recomputed every time
            # because they quote this month's numbers.
            key = f"what|{t}|{SOURCE}"
            what = cache.get(key)
            if what is None:
                what = _yahoo_summary(t) if SOURCE == "yahoo" else ""
                if not what:
                    what = _metadata_sentence(t, name, m)
                cache[key] = what
                dirty = True

            out[t] = {
                "industry": m.get("industry") or m.get("sicindustry") or "",
                "what": what,
                "standing": _standing(h),
                "why": _why(h, i + 1, cfg),
            }
        if dirty:
            # Bound the file. Keyed by ticker, so it grows with the number of
            # distinct names ever held -- slow, but not bounded by anything.
            if len(cache) > 5000:
                cache = dict(sorted(cache.items())[-2500:])
            _save_cache(cache)
    return out
