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

@app.post("/api/portfolio")
def api_portfolio():
    overrides = request.get_json(silent=True) or {}
    return jsonify(engine.compute_portfolio(overrides))


@lru_cache(maxsize=128)
def _cached_backtest(key: str, window: str, hold: str):
    overrides = json.loads(key) if key else {}
    return engine.compute_backtest(overrides, window=window, hold=hold)


@app.post("/api/backtest")
def api_backtest():
    body = request.get_json(silent=True) or {}
    window = (body.pop("window", "MAX") or "MAX")
    hold = (body.pop("hold", "1M") or "1M")
    key = json.dumps(body, sort_keys=True)
    return jsonify(_cached_backtest(key, window, hold))


@lru_cache(maxsize=128)
def _cached_edge_backtest(window: str, hold: int, gate: float):
    import edge_lib
    return edge_lib.run_edge_backtest(window=window, spec={"hold": hold, "rev_growth_gate": gate})


@app.post("/api/edge_backtest")
def api_edge_backtest():
    body = request.get_json(silent=True) or {}
    window = (body.get("window", "MAX") or "MAX")
    hold = int(body.get("hold", 42) or 42)        # 21/42/63/126 = 1M/2M/3M/6M clock
    gate = float(body.get("gate", 0.0) or 0.0)    # optional YoY rev-growth gate (0 = off)
    return jsonify(_cached_edge_backtest(window, hold, gate))


@app.get("/api/edge_tracker")
def api_edge_tracker():
    import edge_tracker_lib
    return jsonify(edge_tracker_lib.tracker_state())


@app.get("/api/stock_vs_sp500")
def api_stock_vs_sp500():
    ck = request.args.get("company_key", "")
    return jsonify(engine.stock_vs_sp500(ck))


@app.get("/api/equations")
def api_equations():
    return jsonify(engine.equations_schema())


@app.get("/api/meta")
def api_meta():
    return jsonify(engine.meta())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
