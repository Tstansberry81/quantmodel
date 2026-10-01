"""The Edge — short-term trading product (local site).

Serves the Edge backtest and the Edge Tracker. All data is read through the
self-contained edge_data layer (cached prices, benchmarks, artifacts); the model
lives in edge_lib.

Run:  python app.py     ->  http://127.0.0.1:5000
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

# The artifact is a pandas-bearing pickle. Unpickling pandas objects walks deep
# __reduce__/__setstate__ chains, and the depth differs across Python/pandas
# versions -- the build machine and the host need not agree. Production hit
# "RecursionError: maximum recursion depth exceeded" on a payload that
# unpickles fine locally, so give the interpreter headroom before anything
# imports pandas. 1000 (the default) is the only thing that was ever tight.
sys.setrecursionlimit(20000)
import threading
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
import live_tracker         # /api/diag reports where its state file lands
import export_vision        # imports edge_lib/edge_tracker_lib too — same reason


import config   # noqa: E402  -- importing config loads .env for every entrypoint

# INFO to stdout. Flask's app.logger and every library logger default to
# WARNING outside debug mode, so edge_lib's "panel cache is stale; rebuilding"
# and "cache warm done in Ns" were never emitted. Their absence from the Render
# logs looked like "no rebuild happened" and was actually "no logging happened"
# -- I spent a long time today reading that silence as evidence.
logging.basicConfig(
    level=os.environ.get("EDGE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout, force=True)

app = Flask(__name__)
app.json.sort_keys = False
app.logger.setLevel(logging.INFO)


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
@lru_cache(maxsize=1)
def _edge_windows() -> frozenset:
    # lazy so the heavy edge_lib import stays off the boot path; the window
    # list itself is single-sourced from edge_lib.WINDOWS
    import edge_lib
    return frozenset(edge_lib.WINDOWS)


_EDGE_HOLDS = {21, 42, 63, 126}
# The selector default is the SHIPPED clock, read from the spec rather than typed
# in. It was hardcoded to 42 in four places; when the model moved to a monthly
# rebalance (2026-07-30) each of those would have kept serving a 2-month backtest
# as "the model" to anyone who hadn't touched the buttons.
_DEFAULT_HOLD = edge_lib.EDGE_SPEC["hold"]
_EDGE_MIXES = {0.0, 0.25, 0.5, 0.75, 1.0}   # growth-mix selector options
_EDGE_NS = set(range(5, 11))                # basket-size selector: 5..10 stocks
# Per-IP bucket for the compute endpoints. The caches (above) bound TOTAL work;
# this bounds the RATE at which a cold cache can be walked, so one client can't
# monopolise the single worker. Generous enough that real UI clicking never trips it.
_bt_hits: dict[str, list] = defaultdict(list)
_BT_RATE, _BT_WINDOW = 30, 60               # 30 requests / minute / IP
# The PRODUCT is a 10-stock book (edge_lib.EDGE_SPEC / edge_tracker_lib.N); the
# selector is for exploration only. Tighter baskets scored better in-sample and
# then failed out of sample (n=5: -57% drawdown), so 10 is the shipped size.


def _safe(fn):
    """Always return JSON, even on error — so the frontend never receives an empty
    body (which would surface as 'Unexpected end of JSON input')."""
    try:
        return jsonify(fn())
    except Exception as e:                       # noqa: BLE001
        app.logger.exception("API error")
        return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"})


# Only ONE heavy computation at a time, process-wide.
#
# gunicorn runs --threads 4, and nothing below the request handlers takes a lock
# except the panel build. So four cold requests each built their own daily
# series concurrently -- four copies of the most memory-hungry computation in
# the process -- and OOM-killed the 2Gi instance three times on 2026-07-30
# (18:09, 18:14, 18:19), the last one 28 seconds after a deploy went live.
#
# This costs nothing real. The work is CPU-bound on a single-worker plan, so
# concurrency was never buying throughput -- the caches are. And because
# lru_cache returns a HIT without entering the function body, cache hits never
# touch this lock; only misses serialize. A warm page load is unaffected.
_COMPUTE_LOCK = threading.Lock()
# Results published by whichever thread actually did the work, keyed by args.
# lru_cache does NOT dedupe in-flight calls: it only stores a result once the
# wrapped call RETURNS. So the boot warm-up thread enters the body and computes
# for minutes while a request for the SAME parameters also enters the body, waits
# on the lock, and then recomputes the identical result. Queued requests were
# doing redundant work instead of collecting the first thread's answer -- the
# same double-checked-locking gap already fixed in load_edge_panel, one layer up.
_COMPUTE_MEMO: dict = {}
# How long to WAIT for a turn. Long enough to outlast a cold boot warm-up (which
# holds the lock for minutes on a small instance) so a queued request wakes up,
# finds the memo populated and returns instantly. Bounded so threads are never
# held indefinitely -- /api/meta is the Render healthCheckPath, and a starved
# health check gets the instance restarted.
_COMPUTE_WAIT_S = 240


def _computed(key, fn):
    """Run `fn` once per key, process-wide, and share the result.

    Three properties that matter, in order:
      1. a HIT never takes the lock (checked before acquiring)
      2. a thread that waited gets the winner's RESULT, not a second computation
      3. waiting is bounded, so a slow compute cannot starve the health check
    """
    hit = _COMPUTE_MEMO.get(key)
    if hit is not None:
        return hit
    if not _COMPUTE_LOCK.acquire(timeout=_COMPUTE_WAIT_S):
        raise TimeoutError(
            "the server is still warming up (one heavy computation runs at a "
            "time, deliberately). Retry in a few seconds.")
    try:
        hit = _COMPUTE_MEMO.get(key)          # someone finished while we waited
        if hit is not None:
            return hit
        val = fn()
        if len(_COMPUTE_MEMO) > 64:           # bounded; lru_cache above is the real cache
            _COMPUTE_MEMO.clear()
        _COMPUTE_MEMO[key] = val
        return val
    finally:
        _COMPUTE_LOCK.release()


# Public selector space = windows(7) x holds(4) x mixes(5) x n(6) = 840 combos.
# Cache ABOVE that so a client cycling parameters can never evict-and-recompute:
# each combo is computed at most once per worker.
#
# SAFE HERE because the entries really are small: a backtest payload measured
# 12 KB (curves are subsampled to ~300 points), so 1024 of them is ~12 MB.
# See _cached_tracker for the same reasoning applied to a payload 40x larger,
# where "never evict" was not affordable.
@lru_cache(maxsize=1024)
def _cached_edge_backtest(window: str, hold: int, mix: float, n: int):
    # clock_spec, not a bare hold. The grid is calendar-anchored now, so passing
    # only `hold` would leave rebal_months at the shipped value and report a
    # monthly book on (say) a 3-month annualization: one number in, a coherent
    # pair out.
    return _computed(
        ("bt", window, hold, mix, n),
        lambda: edge_lib.run_edge_backtest(
            window=window,
            spec={**edge_lib.clock_spec(hold), "growth_mix": mix, "n": n}))


def _warm_caches():
    """Precompute the DEFAULT product backtest (+ tracker) on boot so the first
    real visitor is served from cache instead of eating the ~30s cold build (panel
    + correlation-cap selection). Each gunicorn worker has its own in-process
    lru_cache, so this runs once per worker — exactly what's needed. All windows
    reuse the same cached daily series, so warming MAX warms every window."""
    import time as _t
    t0 = _t.time()
    try:
        _cached_edge_backtest("MAX", _DEFAULT_HOLD, 0.0, 10)  # the default /edge view
        _cached_tracker(_DEFAULT_HOLD, "MAX", 0.0, 10)       # tracker shares the panels
        app.logger.info("cache warm done in %.0fs", _t.time() - t0)
    except Exception:
        app.logger.exception("cache warm failed")


# WARM AFTER THE FORK, NEVER AT IMPORT.
#
# This used to start the warm thread at module import. gunicorn imports the app
# and then FORKS its worker, and a fork copies the lock in whatever state it held
# at that instant while NOT copying the thread that held it. The warm thread took
# _COMPUTE_LOCK, the fork happened, and the child inherited a lock that was held
# by a thread which did not exist there. Nothing could ever release it: every
# compute request waited the full timeout and failed, forever, on a healthy
# instance with a correctly matching panel.
#
# That is what /api/diag finally showed -- compute_lock_held=True with only two
# live threads, neither of them holding it. It is also why "Your service is live"
# preceded "Detected service running on port 10000" by five minutes.
#
# Starting it from the first request guarantees we are past the fork, and works
# identically under `python app.py`.
_WARM_LOCK = threading.Lock()
_WARM_STARTED = False


@app.before_request
def _warm_once():
    """Kick the boot warm-up on the first request, post-fork."""
    global _WARM_STARTED
    if _WARM_STARTED or os.environ.get("EDGE_WARM_ON_BOOT", "1") != "1":
        return
    with _WARM_LOCK:
        if _WARM_STARTED:
            return
        _WARM_STARTED = True
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
    if window not in _edge_windows():
        window = "MAX"
    try:
        hold = int(body.get("hold", _DEFAULT_HOLD))
    except (TypeError, ValueError):
        hold = _DEFAULT_HOLD
    if hold not in _EDGE_HOLDS:
        hold = _DEFAULT_HOLD                     # 21/42/63/126 = 1M/2M/3M/6M clock
    try:
        mix = round(float(body.get("mix", 0.0)), 2)
    except (TypeError, ValueError):
        mix = 0.0
    if mix not in _EDGE_MIXES:
        mix = 0.0        # RETIRED gate -- default OFF (ranked negatively on
                         # survivorship-free data); selectable for exploration only
    n = _parse_n(body.get("n", 10))
    return _safe(lambda: _cached_edge_backtest(window, hold, mix, n))


# NOT 1024. This copied _cached_edge_backtest's "never evict the 840-combo
# space" reasoning, which was sound for a 12 KB payload and ruinous here: a
# tracker payload is 489 KB -- it embeds the whole backtest-seeded log, one entry
# per rebalance, each carrying its 10 holdings and their returns. At 1024 that is
# ~500 MB of JSON, and considerably more as live Python dicts, on a 2 GB box.
#
# The monthly clock is what tipped it: 164 rebalances became 330, doubling the
# payload under a cache size chosen when it was half as big. Two OOM kills on
# 2026-07-30 (server_failed / oomKilled, memoryLimit 2Gi) traced back here.
#
# 12 covers the realistic working set -- a visitor toggling window and basket
# size on one page -- for ~6 MB. A cold recompute costs ~3s, which is the right
# trade against a 2 GB ceiling.
@lru_cache(maxsize=12)
def _cached_tracker(hold: int, window: str, mix: float, n: int):
    return _computed(
        ("tr", hold, window, mix, n),
        lambda: edge_tracker_lib.tracker_state(
            hold=hold, window=window, mix=mix, n=n, finalize=False))


@app.get("/api/edge_tracker")
def api_edge_tracker():
    if not _rate_ok(_client_ip(), _BT_RATE, _BT_WINDOW, _bt_hits):
        return jsonify({"ok": False, "reason": "Too many requests — please wait a moment."})
    try:
        hold = int(request.args.get("hold", _DEFAULT_HOLD))
    except (TypeError, ValueError):
        hold = _DEFAULT_HOLD
    if hold not in _EDGE_HOLDS:
        hold = _DEFAULT_HOLD                     # 21/42/63/126 = 1M/2M/3M/6M clock
    window = (request.args.get("window", "MAX") or "MAX").upper()
    if window not in _edge_windows():
        window = "MAX"
    try:
        mix = round(float(request.args.get("mix", 0.0)), 2)
    except (TypeError, ValueError):
        mix = 0.0
    if mix not in _EDGE_MIXES:
        mix = 0.0        # RETIRED gate -- default OFF (ranked negatively on
                         # survivorship-free data); selectable for exploration only
    n = _parse_n(request.args.get("n", 10))
    # The heavy payload is cached for the process's lifetime; the forward ledger
    # is not. finalize() rescores every forward row from prices on each request
    # (prices themselves are cached ~15 min), so the forward record and the
    # live mark can never again freeze at whatever the process first computed.
    return _safe(lambda: edge_tracker_lib.finalize(_cached_tracker(hold, window, mix, n)))


def _state_paths():
    """Durability of the files that hold unrecoverable state.

    `ephemeral` is the flag that matters: it means the path sits under the
    deploy directory and will be wiped by the next push. Both of these hold
    data no rebuild can reproduce -- the forward record's logged_at timestamps
    and the live mark's ENTRY PRICES -- so neither has a recovery path once it
    is gone. Reported rather than assumed, because the failure is silent.
    """
    import pathlib as _pl
    src = _pl.Path(__file__).resolve().parent
    out = {}
    for name, path in (("forward_record", edge_tracker_lib.SNAPSHOT_PATH),
                       ("live_mark", live_tracker.STATE_PATH)):
        p = _pl.Path(path)
        rec = {"path": str(p), "exists": p.exists()}
        try:
            rec["ephemeral"] = p.resolve().is_relative_to(src)
        except Exception:                                # noqa: BLE001
            rec["ephemeral"] = None
        if p.exists():
            rec["size_kb"] = round(p.stat().st_size / 1e3, 1)
        out[name] = rec
    return out


@app.get("/api/diag")
def api_diag():
    """Why is the panel being rebuilt? Answers it directly instead of by inference.

    Reports the fingerprint production COMPUTES against the one each cached panel
    STORES. A mismatch is the difference between a 0.3s unpickle and a ~4min
    rebuild, and it is otherwise invisible: a stale panel is a cache miss, not an
    error. Cheap and lock-free, so it works while a compute is in flight."""
    import pickle as _pk
    clock = edge_lib.clock_spec(edge_lib.EDGE_SPEC["hold"])
    want = edge_lib._panel_fingerprint(clock["hold"], edge_lib.UNIVERSE, 0,
                                       clock["rebal_months"])
    reads, write = edge_lib._panel_cache_paths(clock["hold"], edge_lib.UNIVERSE, 0,
                                               clock["rebal_months"])
    panels = []
    for p in reads:
        rec = {"path": str(p), "exists": p.exists()}
        if p.exists():
            rec["size_mb"] = round(p.stat().st_size / 1e6, 1)
            try:
                with open(p, "rb") as fh:
                    rec["fingerprint"] = _pk.load(fh).get("fingerprint")
                rec["MATCHES"] = rec.get("fingerprint") == want
            except Exception as e:                       # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
        panels.append(rec)
    return jsonify({
        "ok": True,
        "expected_fingerprint": want,
        "panels": panels,
        "knobs": {"use_pit_universe": edge_lib.USE_PIT_UNIVERSE,
                  "delist_haircut": edge_lib.DELIST_HAIRCUT,
                  "lb": edge_lib.LB, **clock,
                  "universe": edge_lib.UNIVERSE},
        "panel_cache_dir": str(edge_lib.PANEL_CACHE_DIR),
        # WHERE does state that cannot be rebuilt actually live, and is it there?
        # A file on ephemeral storage looks identical to a healthy one from the
        # outside -- it just quietly restarts on every deploy, and the page reads
        # correct for a run one deploy old. That is how the live mark spent a day
        # resetting to "day 1" with the disk mounted and working the whole time.
        "state": _state_paths(),
        "compute_lock_held": _COMPUTE_LOCK.locked(),
        "memo_keys": [str(k) for k in _COMPUTE_MEMO],
        # WHERE is the lock holder? A held lock with an empty memo means a thread
        # is inside fn() and has been for a while; the only way to know which
        # line without guessing is to look at every thread's stack. Trimmed to
        # our own frames so the output stays readable.
        "threads": _thread_stacks(),
    })


def _thread_stacks() -> dict:
    import sys as _s, traceback, threading as _th
    names = {t.ident: t.name for t in _th.enumerate()}
    out = {}
    for tid, frame in _s._current_frames().items():
        stack = traceback.extract_stack(frame)
        ours = [f"{f.filename.rsplit('/',1)[-1]}:{f.lineno} {f.name}"
                for f in stack
                if any(m in f.filename for m in ("app.py", "edge_lib", "edge_data",
                                                 "edge_tracker_lib", "export_vision"))]
        out[names.get(tid, str(tid))] = ours[-6:] or [f"{stack[-1].filename.rsplit('/',1)[-1]}:"
                                                      f"{stack[-1].lineno} {stack[-1].name}"]
    return out


@app.get("/api/meta")
def api_meta():
    return _safe(engine.meta)


@app.post("/api/sync_vision")
def api_sync_vision():
    """Push the CURRENT backtest settings (window / rebalance clock, 10-stock book)
    to the Vision product via the export pipeline + a GitHub commit (falls back to
    a local write with no GITHUB_TOKEN)."""
    body = request.get_json(silent=True) or {}
    window = (body.get("window", "2Y") or "2Y").upper()
    if window not in _edge_windows():
        window = "2Y"
    # Default to the SHIPPED clock, not 21. Defaulting to a hold the product does
    # not use meant a sync with no explicit hold published a different model to a
    # public site (the growth-mix argument, now removed, did the same).
    _default_hold = _DEFAULT_HOLD
    try:
        hold = int(body.get("hold", _default_hold))
    except (TypeError, ValueError):
        hold = _default_hold
    if hold not in _EDGE_HOLDS:
        hold = _default_hold
    return _safe(lambda: export_vision.export_to_vision(window=window, hold=hold))


# ---- shared request helpers ------------------------------------------------
def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "?"

def _rate_ok(ip: str, limit: int, window: int, hits: dict) -> bool:
    """Sliding-window per-IP limiter over the caller's bucket.

    `hits` used to default to the chat assistant's bucket. With the chat removed
    a default would silently share one bucket between unrelated endpoints, so it
    is now required — the caller states which limiter it means."""
    now = time.time()
    q = hits[ip]
    while q and q[0] < now - window:
        q.pop(0)
    if len(q) >= limit:
        return False
    q.append(now)
    return True


if __name__ == "__main__":
    # macOS squats :5000 with the AirPlay Receiver, so the port is worth being
    # able to set without touching the environment.
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    args = ap.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=True, use_reloader=False)
