"""The Edge — short-term trading product (local site).

The long-term "Slow Burn" model + curated Portfolio now live in their own
codebase ("personal quant model"). This app serves the Edge backtest and the
Edge Tracker. The qmodel package is retained as a library (the Edge reuses its
cached-data loaders + benchmarks via edge_lib / tech_bias_lib).

Run:  python app.py     ->  http://127.0.0.1:5000
"""
from __future__ import annotations
import os
from functools import lru_cache

from flask import Flask, render_template, request, jsonify, redirect

from qmodel import engine   # cached-data loaders + meta used by the Edge + footer

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
