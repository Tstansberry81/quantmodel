"""Three-sentence company write-ups for the stocks in the Vision book.

WHAT IT PRODUCES, per holding:
  industry   -- Sharadar's own label ("Semiconductors"). NOT generated.
  what       -- what the business actually does
  different  -- what separates it from the obvious comparison
  why        -- why THIS MODEL bought it

WHY THE PROSE IS GENERATED
Sharadar ships sector, industry, location and a company website; it ships no
business description. Nothing in the artifact can be templated into a readable
sentence, so the sentences are written by Claude from the facts we do hold.

THE FAILURE MODE THIS FILE IS BUILT AROUND
"Why we are buying it" is the sentence most likely to become fiction. A language
model asked why a fund owns NVDA will happily produce a moat-and-TAM thesis. The
Edge has no opinion about moats: it buys 12-1 momentum, above a $10B floor,
capped at 2 names per sector, de-levered by a vol target. So the prompt is handed
the REAL numbers behind the pick and instructed to explain those and nothing
else -- and the site labels the text as model-written.

Prose can never move a number. Everything here is display text hanging off a book
that was already chosen; if generation fails the book publishes without it.

CACHING
Keyed on (ticker, book_date, prompt+model fingerprint) and stored as JSON. Three
consequences worth stating:
  * A repeated export costs nothing. "Sync to Vision" is a button; without a
    cache every click would re-bill and re-latency ten LLM calls.
  * A new name entering the book is the ONLY thing that triggers generation, so
    the pipeline stays automatic with no list to maintain by hand.
  * book_date is in the key ON PURPOSE. The `why` sentence quotes live momentum
    numbers, so a cache keyed on ticker alone would still be reciting last
    quarter's figures a year later. Regenerating monthly means the wording of the
    first two sentences can drift slightly between rebalances; that is the
    accepted price of never publishing a stale number.

The fingerprint hashes the prompt text and the model name rather than a
hand-bumped version constant -- the same reasoning as edge_lib's panel
fingerprint: "remember to bump the version" is a rule that gets forgotten once
and then lies forever.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import pathlib
import threading

import config

log = logging.getLogger(__name__)

MODEL = os.environ.get("EDGE_DESC_MODEL", "claude-sonnet-4-6")
CACHE_PATH = pathlib.Path(os.environ.get(
    "EDGE_DESC_PATH", str(config.CACHE_DIR / "company_desc.json")))
META_PATH = config.ARTIFACT_DIR / "company_meta.json"

# One writer at a time. gunicorn runs --threads 4 and two clicks of "Sync to
# Vision" can land together; a read-modify-write of one JSON file from two
# threads loses one thread's descriptions.
#
# The lock is deliberately held ACROSS the API call, not just the file write.
# That serializes concurrent exports -- the second click waits rather than
# billing a duplicate generation for the same book, and then finds the first
# click's results already in the cache. The cost is that a second click blocks
# for the length of one request; acceptable for a manual button, and much better
# than two threads generating the same ten descriptions and one of them losing
# the write.
_LOCK = threading.Lock()

_SYSTEM = """You write short factual company blurbs for a quantitative stock \
newsletter. You will be given real metadata for companies a momentum model just \
bought, and the model's actual reason for buying each one.

Rules:
- Exactly three fields per company: "what", "different", "why". One sentence each.
- "what": what the business does and how it makes money. Concrete and specific.
- "different": what separates it from its most obvious competitor or from the \
rest of its industry. If you do not know something genuinely distinguishing, \
describe its position in the industry instead of inventing a moat.
- "why": the MODEL's reason, and ONLY the model's reason. You are given the \
signal values -- restate what they mean in plain English. Never invent a \
fundamental, valuation, product-cycle or macro thesis; this model does not read \
financial statements and has no view on the company's future.
- No hype, no adjectives like "leading" or "revolutionary", no price targets, no \
recommendations, no forecasts.
- If you are not confident what a company does, say so plainly in "what" rather \
than guessing. A blank is better than a wrong fact.
- Under 30 words per sentence.

Reply with ONLY a JSON object mapping each ticker to {"what","different","why"}. \
No markdown fences, no commentary."""


def _fingerprint() -> str:
    return hashlib.sha256(f"{_SYSTEM}\x00{MODEL}\x00v1".encode()).hexdigest()[:12]


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt cache must never break an export -- regenerating always works.
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


def _facts(holding: dict, meta: dict, signal_label: str) -> dict:
    """The grounding facts handed to the model. Deliberately small: every field
    here is something we actually hold, and the prompt forbids going beyond it."""
    t = str(holding.get("ticker") or "").upper()
    m = meta.get(t, {})
    f = {"ticker": t,
         "company_name": holding.get("name") or m.get("name") or t,
         "sector": holding.get("sector") or "",
         "industry": m.get("industry") or m.get("sicindustry") or "",
         "headquarters": m.get("location") or "",
         "website": m.get("companysite") or ""}
    sig = holding.get("signal")
    if sig is not None:
        # The signal IS the reason. Stated as a percentage because that is what
        # ret_12_1 is -- a trailing return, not a score.
        f["model_signal"] = (f"{signal_label} = {float(sig) * 100:+.1f}%"
                             if abs(float(sig)) < 20 else f"{signal_label} = {sig}")
    if holding.get("weight") is not None:
        f["weight_in_book"] = f"{float(holding['weight']) * 100:.0f}%"
    if holding.get("rank") is not None:
        f["rank_in_book"] = holding["rank"]
    return {k: v for k, v in f.items() if v not in ("", None)}


def _generate(missing: list[dict], signal_label: str, model_rules: str) -> dict:
    """One call for the whole book, so the model can differentiate the names from
    each other in the "different" sentence. Returns {ticker: {...}}; tickers it
    fails to return are simply left out and the caller degrades gracefully."""
    import anthropic
    client = anthropic.Anthropic()
    payload = {"how_this_model_picks": model_rules, "companies": missing}
    resp = client.messages.create(
        model=MODEL, max_tokens=2000, system=_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, indent=1)}])
    txt = "".join(b.text for b in resp.content if b.type == "text").strip()
    if txt.startswith("```"):                     # strip a fence if one appears
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    out = json.loads(txt)
    clean = {}
    for t, v in (out or {}).items():
        if isinstance(v, dict):
            clean[str(t).upper()] = {k: str(v.get(k) or "").strip()
                                     for k in ("what", "different", "why")}
    return clean


def describe(book: list[dict], book_date: str, signal_label: str = "12-1 momentum",
             model_rules: str = "") -> dict:
    """{ticker: {industry, what, different, why}} for every holding in `book`.

    Always returns an entry per holding, even with no API key or a failed call --
    `industry` comes from Sharadar and needs no model, so the page can show the
    category and simply omit the prose. Never raises: a description problem must
    not be able to block publishing the book.
    """
    meta = _company_meta()
    fp = _fingerprint()
    out, missing = {}, []

    with _LOCK:
        cache = _load_cache()
        for i, h in enumerate(book):
            t = str(h.get("ticker") or "").upper()
            if not t:
                continue
            out[t] = {"industry": meta.get(t, {}).get("industry")
                      or meta.get(t, {}).get("sicindustry") or ""}
            hit = cache.get(f"{t}|{book_date}|{fp}")
            if hit:
                out[t].update(hit)
            else:
                missing.append(_facts({**h, "rank": i + 1}, meta, signal_label))

        if missing and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                got = _generate(missing, signal_label, model_rules)
            except Exception as e:                          # noqa: BLE001
                log.warning("description generation failed: %s: %s",
                            type(e).__name__, e)
                got = {}
            for t, v in got.items():
                if t in out and any(v.values()):
                    out[t].update(v)
                    cache[f"{t}|{book_date}|{fp}"] = v
            if got:
                # Bound the file: this accrues one entry per (name, rebalance)
                # forever otherwise, on a 1GB disk shared with the paper record.
                # Evict by BOOK DATE, not by key order -- keys start with the
                # ticker, so a plain sort would throw away every name beginning
                # with A and keep the Zs, which is not an eviction policy.
                if len(cache) > 4000:
                    cache = dict(sorted(cache.items(),
                                        key=lambda kv: kv[0].split("|")[1:2])[-2000:])
                _save_cache(cache)
        elif missing:
            log.info("no ANTHROPIC_API_KEY — publishing %d holdings without prose",
                     len(missing))
    return out
