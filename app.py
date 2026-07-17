"""The Edge — short-term trading product (local site).

Serves the Edge backtest and the Edge Tracker. All data is read through the
self-contained edge_data layer (cached prices, benchmarks, artifacts); the model
lives in edge_lib.

Run:  python app.py     ->  http://127.0.0.1:5000
"""
from __future__ import annotations
import os
import time
from collections import defaultdict
from functools import lru_cache

from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.exceptions import HTTPException

import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)
# Import the model modules eagerly, at module scope. These MUST NOT be imported
# lazily inside request handlers: the boot warm-up thread and an incoming request
# would then race to `import edge_lib`, and whichever loses grabs the half-built
# module out of sys.modules -> "partially initialized module ... has no attribute
# run_edge_backtest". Importing here completes the import once, single-threaded,
# before the warm thread starts. (Module import is cheap; the heavy panel build
# stays lazy behind lru_cache inside edge_lib.)
import edge_lib
import edge_tracker_lib
import export_sauron        # imports edge_lib/edge_tracker_lib too — same reason


def _load_dotenv():
    """Load KEY=VALUE pairs from a local .env (gitignored) into the environment.
    Used for ANTHROPIC_API_KEY in local dev; on Render set it as a real env var."""
    p = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

app = Flask(__name__)
app.json.sort_keys = False


# ---- error handlers --------------------------------------------------------
# Guarantee that anything under /api/ always answers with JSON — never an HTML
# error page — for EVERY status code, not just 404/405/500. Without this, an
# error returns Flask's HTML page and the frontend's `await r.json()` blows up
# with the cryptic "Unexpected token '<', "<!DOCTYPE"... is not valid JSON".

def _wants_json() -> bool:
    return request.path.startswith("/api/")


@app.errorhandler(HTTPException)
def _err_http(e):
    """Any HTTP error (404, 405, 400, 413, 503, …) → JSON on /api/ paths;
    otherwise Flask's normal HTML error page. One handler covers all codes so
    the always-JSON invariant can't be missed for an un-enumerated status."""
    if _wants_json():
        return jsonify({"ok": False, "reason": e.description or e.name}), (e.code or 500)
    return e


@app.errorhandler(Exception)
def _err_any(e):
    """Unexpected non-HTTP exception: JSON 500 on /api/ paths, otherwise re-raise
    so Flask's normal 500 page (and the dev debugger) handles it."""
    if isinstance(e, HTTPException):     # safety net; normally routed to _err_http
        return _err_http(e)
    app.logger.exception("Unhandled error")
    if _wants_json():
        return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"}), 500
    raise e


# ---- pages -----------------------------------------------------------------

@app.route("/")
def home():
    return redirect("/edge")


@app.route("/edge")
def page_edge():
    return render_template("edge.html")


@app.route("/edge-tracker")
def page_edge_tracker():
    return render_template("edge_tracker.html")


@app.route("/model")
def page_model():
    return render_template("model.html")


# ---- json api --------------------------------------------------------------
_EDGE_WINDOWS = {"1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "MAX"}
_EDGE_HOLDS = {21, 42, 63, 126}
_EDGE_MIXES = {0.0, 0.25, 0.5, 0.75, 1.0}   # growth-mix selector options
_EDGE_NS = set(range(5, 11))                # basket-size selector: 5..10 stocks
# Per-IP bucket for the compute endpoints. The caches (above) bound TOTAL work;
# this bounds the RATE at which a cold cache can be walked, so one client can't
# monopolise the single worker. Generous enough that real UI clicking never trips it.
_bt_hits: dict[str, list] = defaultdict(list)
_BT_RATE, _BT_WINDOW = 30, 60               # 30 requests / minute / IP
# The PRODUCT stays a 10-stock book (edge_lib.EDGE_SPEC / edge_tracker_lib.N);
# the selector is for exploration. n=7 is the research candidate (2026-07 signal
# hunt) and is paper-tracked forward alongside the product in the tracker.


def _safe(fn):
    """Always return JSON, even on error — so the frontend never receives an empty
    body (which would surface as 'Unexpected end of JSON input')."""
    try:
        return jsonify(fn())
    except Exception as e:                       # noqa: BLE001
        app.logger.exception("API error")
        return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"})


# Public selector space = windows(7) x holds(4) x mixes(5) x n(6) = 840 combos.
# Cache ABOVE that so a client cycling parameters can never evict-and-recompute:
# each combo is computed at most once per worker. (Entries are result dicts with
# small curves, ~tens of KB -> ~25MB fully populated.)
@lru_cache(maxsize=1024)
def _cached_edge_backtest(window: str, hold: int, mix: float, n: int):
    return edge_lib.run_edge_backtest(window=window,
                                      spec={"hold": hold, "growth_mix": mix, "n": n})


def _warm_caches():
    """Precompute the DEFAULT product backtest (+ tracker) on boot so the first
    real visitor is served from cache instead of eating the ~30s cold build (panel
    + correlation-cap selection). Each gunicorn worker has its own in-process
    lru_cache, so this runs once per worker — exactly what's needed. All windows
    reuse the same cached daily series, so warming MAX warms every window."""
    import time as _t
    t0 = _t.time()
    try:
        _cached_edge_backtest("MAX", 42, 0.75, 10)          # the default /edge view
        edge_tracker_lib.tracker_state()                    # tracker shares the panels
        app.logger.info("cache warm done in %.0fs", _t.time() - t0)
    except Exception:
        app.logger.exception("cache warm failed")


if os.environ.get("EDGE_WARM_ON_BOOT", "1") == "1":
    import threading
    threading.Thread(target=_warm_caches, name="edge-warm", daemon=True).start()


def _parse_n(raw) -> int:
    """Basket size 5..10; anything else falls back to the 10-stock product book.
    (Below 5 a basket is single-name risk and the backtest is noise-dominated.)"""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 10
    return n if n in _EDGE_NS else 10


@app.post("/api/edge_backtest")
def api_edge_backtest():
    if not _rate_ok(_client_ip(), _BT_RATE, _BT_WINDOW, _bt_hits):
        return jsonify({"ok": False, "reason": "Too many requests — please wait a moment."})
    body = request.get_json(silent=True) or {}
    window = (body.get("window", "MAX") or "MAX").upper()
    if window not in _EDGE_WINDOWS:
        window = "MAX"
    try:
        hold = int(body.get("hold", 42))
    except (TypeError, ValueError):
        hold = 42
    if hold not in _EDGE_HOLDS:
        hold = 42                                # 21/42/63/126 = 1M/2M/3M/6M clock
    try:
        mix = round(float(body.get("mix", 0.75)), 2)
    except (TypeError, ValueError):
        mix = 0.75
    if mix not in _EDGE_MIXES:
        mix = 0.75                               # growth-mix: fraction of basket from >=15% growth pool
    n = _parse_n(body.get("n", 10))
    return _safe(lambda: _cached_edge_backtest(window, hold, mix, n))


@lru_cache(maxsize=1024)            # same 840-combo space — never evict (see above)
def _cached_tracker(hold: int, window: str, mix: float, n: int):
    return edge_tracker_lib.tracker_state(hold=hold, window=window, mix=mix, n=n)


@app.get("/api/edge_tracker")
def api_edge_tracker():
    if not _rate_ok(_client_ip(), _BT_RATE, _BT_WINDOW, _bt_hits):
        return jsonify({"ok": False, "reason": "Too many requests — please wait a moment."})
    try:
        hold = int(request.args.get("hold", 42))
    except (TypeError, ValueError):
        hold = 42
    if hold not in _EDGE_HOLDS:
        hold = 42                                # 21/42/63/126 = 1M/2M/3M/6M clock
    window = (request.args.get("window", "MAX") or "MAX").upper()
    if window not in _EDGE_WINDOWS:
        window = "MAX"
    try:
        mix = round(float(request.args.get("mix", 0.75)), 2)
    except (TypeError, ValueError):
        mix = 0.75
    if mix not in _EDGE_MIXES:
        mix = 0.75                               # growth-mix: fraction of basket from >=15% growth pool
    n = _parse_n(request.args.get("n", 10))
    return _safe(lambda: _cached_tracker(hold, window, mix, n))


@app.get("/api/meta")
def api_meta():
    return _safe(engine.meta)


@app.post("/api/sync_vision")
def api_sync_vision():
    """Push the CURRENT backtest settings (window / rebalance clock / growth mix,
    10-stock book) to the Vision product via the export pipeline + a GitHub commit
    (falls back to a local write with no GITHUB_TOKEN)."""
    body = request.get_json(silent=True) or {}
    window = (body.get("window", "2Y") or "2Y").upper()
    if window not in _EDGE_WINDOWS:
        window = "2Y"
    try:
        hold = int(body.get("hold", 21))
    except (TypeError, ValueError):
        hold = 21
    if hold not in _EDGE_HOLDS:
        hold = 21
    try:
        mix = round(float(body.get("mix", 0.75)), 2)
    except (TypeError, ValueError):
        mix = 0.75
    if mix not in _EDGE_MIXES:
        mix = 0.75
    return _safe(lambda: export_sauron.export_to_vision(window=window, hold=hold, mix=mix))


# ---- in-app chat assistant ("explain the numbers on screen") ----------------
_CHAT_MODEL = "claude-sonnet-4-6"      # cheap, strong at plain-language explanation
_chat_hits: dict[str, list] = defaultdict(list)
_chat_client = None


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "?"


def _rate_ok(ip: str, limit: int = 20, window: int = 60, hits=None) -> bool:
    """Sliding-window per-IP limiter. `hits` selects the bucket so the chat and the
    compute endpoints are limited independently."""
    now = time.time()
    q = (_chat_hits if hits is None else hits)[ip]
    while q and q[0] < now - window:
        q.pop(0)
    if len(q) >= limit:
        return False
    q.append(now)
    return True


_CHAT_SYSTEM = (
    "You are a concise assistant embedded in a short-term quantitative trading research "
    "dashboard. Answer the user's question about the numbers and data shown on the current "
    "page, in plain, jargon-light language a non-expert can follow.\n\n"
    "RULES:\n"
    "- Ground every answer in the PAGE DATA provided. Do not invent figures that aren't in it; "
    "if the data doesn't contain the answer, say so plainly.\n"
    "- Explain and contextualize; define terms (Sharpe, drawdown, alpha, etc.) simply.\n"
    "- This is research, NOT investment advice — never tell the user to buy or sell anything.\n"
    "- Keep answers short: a few sentences, no preamble.\n"
)


@app.post("/api/chat")
def api_chat():
    if not _rate_ok(_client_ip()):
        return jsonify({"ok": False, "reason": "Too many messages — please wait a moment."})
    body = request.get_json(silent=True) or {}
    msg = (body.get("message") or "").strip()[:2000]
    if not msg:
        return jsonify({"ok": False, "reason": "Empty message."})
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"ok": False, "reason": "Chat isn't configured (no API key set)."})

    page = (body.get("page") or "")[:200]
    context = (body.get("context") or "")[:8000]
    # sanitize the last few turns of history
    hist = []
    for h in (body.get("history") or [])[-6:]:
        role = h.get("role")
        content = str(h.get("content") or "")[:2000]
        if role in ("user", "assistant") and content:
            hist.append({"role": role, "content": content})

    try:
        global _chat_client
        if _chat_client is None:
            import anthropic
            _chat_client = anthropic.Anthropic()
        system = f"{_CHAT_SYSTEM}\nCURRENT PAGE: {page}\n\nPAGE DATA (what's on screen):\n{context}"
        resp = _chat_client.messages.create(
            model=_CHAT_MODEL,
            max_tokens=1024,
            system=system,
            messages=hist + [{"role": "user", "content": msg}],
        )
        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
        return jsonify({"ok": True, "reply": reply or "(no response)"})
    except Exception as e:                       # noqa: BLE001
        app.logger.exception("chat error")
        return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
