"""After the bundle is published: redeploy, wait, log the book, ship to Vision.

The three steps that actually MAKE the rebalance happen, in order, with the
verification each one needs. Publishing a bundle changes nothing on its own --
the site keeps serving the panel it already has until it restarts against the
new artifact.

  1. TRIGGER the deploy. Render's build step runs fetch_data.py, which pulls the
     new bundle; the new backtest_data.pkl changes the panel fingerprint, so the
     cached panel on /var/data is correctly rejected and rebuilt.

  2. WAIT for it, by polling /api/meta until `built_at` MOVES. Not a fixed sleep:
     a build that OOMs or fails leaves the old instance serving happily, and a
     sleep would then hand a green tick to a rebalance that never happened.
     built_at comes from the bundle's own meta.json, so a change to it is proof
     the new data is live -- the one signal that cannot be faked by a restart.

  3. LOG the book, by GETting /api/edge_tracker. This is not a health check: the
     forward-record snapshot is written by _persist_snapshots INSIDE that
     request handler, so until something calls it the rebalance exists in the
     panel and nowhere durable. Then POST /api/sync_vision to publish.

No Render API key is needed. The deploy hook is an unguessable URL and the two
app endpoints are unauthenticated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("QUANT_BASE_URL", "https://quantmodel.onrender.com").rstrip("/")


def _req(url: str, method: str = "GET", body: dict | None = None, timeout: int = 300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"User-Agent": "quant-rebalance-bot",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw[:400]}


def meta() -> dict:
    return _req(f"{BASE}/api/meta", timeout=60)


def trigger_deploy(hook: str) -> None:
    print("deploy: triggering…")
    _req(hook, method="POST", timeout=60)


def wait_for_new_build(before: str, minutes: int) -> str:
    """Poll /api/meta until built_at changes. Returns the new value."""
    deadline = time.time() + minutes * 60
    last_err = ""
    while time.time() < deadline:
        time.sleep(20)
        try:
            now = str(meta().get("built_at", ""))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            # Expected while the instance restarts -- keep waiting, don't fail.
            last_err = str(e)
            continue
        if now and now != before:
            print(f"deploy: live — built_at {before!r} -> {now!r}")
            return now
        print(f"deploy: waiting… built_at still {now!r}")
    raise TimeoutError(
        f"deploy did not go live within {minutes} min (built_at never moved off "
        f"{before!r}; last error: {last_err or 'none'}). The old instance is "
        "still serving the previous bundle — check Render events for oomKilled.")


def log_book(expect_book_date: str = "", phase: str = "") -> dict:
    """GET the tracker, which writes the forward-record snapshot as a side effect."""
    print("tracker: requesting (this writes the forward-record snapshot)…")
    t = _req(f"{BASE}/api/edge_tracker?window=MAX", timeout=300)
    if not t.get("ok"):
        raise RuntimeError(f"tracker returned not-ok: {str(t)[:300]}")
    got = t.get("book_date", "")
    print(f"tracker: book_date={got}")
    if expect_book_date and got != expect_book_date:
        # FATAL, and deliberately so. This means the panel did not advance to the
        # rebalance the gate expected -- almost always because Sharadar had not
        # published the entry bar by the time we pulled, so `live_date` is None
        # and _current_book has fallen back to the PREVIOUS rebalance.
        #
        # The bundle is already published and the site is already serving, which
        # is fine: it is serving the book it legitimately still holds. What must
        # NOT happen is step two -- pushing that stale book to Vision, where it is
        # presented to subscribers as this month's pick. Publishing the wrong book
        # is far more expensive than a red run, and tomorrow's scheduled attempt
        # picks it up at a lag the forward record still scores.
        raise RuntimeError(
            f"book_date is {got!r}, expected {expect_book_date!r}. The panel did "
            "not advance to this month's rebalance (the entry bar is probably not "
            "in the data yet). NOT syncing to Vision — the next scheduled run "
            "will retry.")
    if phase == "price" and t.get("entry_px_pending"):
        # The whole point of the price run. If the entry bar did not make it into
        # the panel, the book stays unpriced and the live mark keeps marking the
        # PREVIOUS basket as if it were current -- the Sep 2026 failure, which
        # passed green and went unnoticed for a month. Fail loudly instead.
        raise RuntimeError(
            f"book {got} is still entry_px_pending after the price run (entry "
            f"bar {t.get('entry_date')!r}). The data does not reach the entry "
            "bar; re-run with force once Sharadar has posted it.")
    fl = t.get("forward_log") or []
    if fl:
        last = fl[-1]
        print(f"tracker: newest forward row {last.get('book_date')} "
              f"status={last.get('status')} lag={last.get('logged_lag_days')}d")
    return t


def sync_vision(window: str = "5Y") -> dict:
    print("vision: syncing…")
    r = _req(f"{BASE}/api/sync_vision", method="POST", body={"window": window},
             timeout=300)
    pushed = r.get("pushed")
    if pushed is False:
        raise RuntimeError(f"vision sync did not push: {r.get('reason')!r}. "
                           "Check VISION_GITHUB_TOKEN on the Render service.")
    print(f"vision: pushed — {json.dumps(r)[:300]}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-date", default="", help="book date the gate expects")
    ap.add_argument("--window", default="5Y", help="window to publish to Vision")
    ap.add_argument("--phase", default="", help="gate phase: publish | price")
    ap.add_argument("--wait-minutes", type=int, default=45)
    ap.add_argument("--skip-deploy", action="store_true")
    args = ap.parse_args()

    hook = os.environ.get("RENDER_DEPLOY_HOOK", "").strip()
    if not hook and not args.skip_deploy:
        print("post_deploy: RENDER_DEPLOY_HOOK is not set", file=sys.stderr)
        return 2

    before = str(meta().get("built_at", ""))
    print(f"deploy: built_at before = {before!r}")

    if not args.skip_deploy:
        trigger_deploy(hook)
        wait_for_new_build(before, args.wait_minutes)

    log_book(args.book_date, args.phase)
    sync_vision(args.window)
    print("post_deploy: done — book logged and Vision updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
