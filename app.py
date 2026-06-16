"""Local website for the quant model.

Run:  python app.py     ->  http://127.0.0.1:5000
"""
from __future__ import annotations
import json
import os
from functools import lru_cache

from flask import Flask, render_template, request, jsonify

from qmodel import engine

app = Flask(__name__)
app.json.sort_keys = False  # preserve logical factor order in JSON responses


# ---- pages -----------------------------------------------------------------

@app.route("/")
def page_portfolio():
    return render_template("portfolio.html")


@app.route("/backtest")
def page_backtest():
    return render_template("backtest.html")


@app.route("/edge")
def page_edge():
    return render_template("edge.html")


@app.route("/edge-tracker")
def page_edge_tracker():
    return render_template("edge_tracker.html")


# ---- json api --------------------------------------------------------------
# Allowed inputs (validated so bad/abusive params can't error or bloat the cache).
_EDGE_WINDOWS = {"1Y", "2Y", "3Y", "5Y", "MAX"}
_EDGE_HOLDS = {21, 42, 63, 126}
_EDGE_GATES = {0.0, 0.05, 0.10, 0.15}
_BT_WINDOWS = {"1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "MAX"}
_BT_HOLDS = {"1W", "2W", "1M", "3M", "6M", "12M"}


def _safe(fn):
    """Always return JSON, even on error — so the frontend never receives an empty
    body (which would surface as 'Unexpected end of JSON input')."""
    try:
        return jsonify(fn())
    except Exception as e:                       # noqa: BLE001
        app.logger.exception("API error")
        return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"})


@app.post("/api/portfolio")
def api_portfolio():
    overrides = request.get_json(silent=True) or {}
    return _safe(lambda: engine.compute_portfolio(overrides))


@lru_cache(maxsize=128)
def _cached_backtest(key: str, window: str, hold: str):
    overrides = json.loads(key) if key else {}
    return engine.compute_backtest(overrides, window=window, hold=hold)


@app.post("/api/backtest")
def api_backtest():
    body = request.get_json(silent=True) or {}
    window = (body.pop("window", "MAX") or "MAX").upper()
    if window not in _BT_WINDOWS:
        window = "MAX"
    hold = (body.pop("hold", "1M") or "1M").upper()
    if hold not in _BT_HOLDS:
        hold = "1M"
    key = json.dumps(body, sort_keys=True)
    return _safe(lambda: _cached_backtest(key, window, hold))


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


@app.get("/api/stock_vs_sp500")
def api_stock_vs_sp500():
    ck = request.args.get("company_key", "")
    return _safe(lambda: engine.stock_vs_sp500(ck))


@app.get("/api/equations")
def api_equations():
    return _safe(engine.equations_schema)


@app.get("/api/meta")
def api_meta():
    return _safe(engine.meta)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
