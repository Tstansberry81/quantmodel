"""The Edge — short-term trading product (local site).

The long-term "Slow Burn" model + curated Portfolio now live in their own
codebase ("personal quant model"). This app serves the Edge backtest and the
Edge Tracker. The qmodel package is retained as a library (the Edge reuses its
cached-data loaders + benchmarks via edge_lib / tech_bias_lib).

Run:  python app.py     ->  http://127.0.0.1:5000
"""
from __future__ import annotations
import os
import time
from collections import defaultdict
from functools import lru_cache

from flask import Flask, render_template, request, jsonify, redirect

from qmodel import engine   # cached-data loaders + meta used by the Edge + footer


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


# ---- json api --------------------------------------------------------------
_EDGE_WINDOWS = {"1Y", "2Y", "3Y", "5Y", "MAX"}
_EDGE_HOLDS = {21, 42, 63, 126}
_EDGE_GATES = {0.0, 0.05, 0.10, 0.15}


def _safe(fn):
    """Always return JSON, even on error — so the frontend never receives an empty
    body (which would surface as 'Unexpected end of JSON input')."""
    try:
        return jsonify(fn())
    except Exception as e:                       # noqa: BLE001
        app.logger.exception("API error")
        return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"})


@lru_cache(maxsize=128)
def _cached_edge_backtest(window: str, hold: int, gate: float):
    import edge_lib
    return edge_lib.run_edge_backtest(window=window, spec={"hold": hold, "rev_growth_gate": gate})


@app.post("/api/edge_backtest")
def api_edge_backtest():
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
        gate = round(float(body.get("gate", 0.0)), 2)
    except (TypeError, ValueError):
        gate = 0.0
    if gate not in _EDGE_GATES:
        gate = 0.0                               # optional YoY rev-growth gate
    return _safe(lambda: _cached_edge_backtest(window, hold, gate))


@app.get("/api/edge_tracker")
def api_edge_tracker():
    import edge_tracker_lib
    return _safe(lambda: edge_tracker_lib.tracker_state())


@app.get("/api/meta")
def api_meta():
    return _safe(engine.meta)


# ---- in-app chat assistant ("explain the numbers on screen") ----------------
_CHAT_MODEL = "claude-sonnet-4-6"      # cheap, strong at plain-language explanation
_chat_hits: dict[str, list] = defaultdict(list)
_chat_client = None


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "?"


def _rate_ok(ip: str, limit: int = 20, window: int = 60) -> bool:
    now = time.time()
    q = _chat_hits[ip]
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
