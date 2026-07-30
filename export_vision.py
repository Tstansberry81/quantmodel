"""Pipeline: export the in-house Edge model to the Vision product site.

This repo (the Edge) is the in-house MODEL + viewer. **Vision** is the consumer
product: a lightweight static site (Tstansberry81/vision) that renders the model's
output. This builds Vision's data file — the current 10-stock book + a primary
backtest curve + a 20-year track-record curve (each vs S&P) + headline stats.

Two ways to run it:
  * CLI:  python export_vision.py [--window 2Y --hold 42]  (writes local)
  * In-app "Sync to Vision" button -> export_to_vision(window, hold), which
    pushes vision_data.js to the Vision GitHub repo (GITHUB_TOKEN) so Render
    auto-redeploys. With no token (dev) it falls back to writing the local file.

Output is JS (window.VISION_DATA = {...}) not JSON so the static site works even
opened directly (file://) with no server/CORS/API key.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from datetime import datetime, timezone

import company_desc
import edge_lib
import edge_tracker_lib

WINDOW = "2Y"          # default primary backtest curve (must be in edge_lib.WINDOWS)
# DERIVED from the shipped spec, never hardcoded. These used to read HOLD = 21
# and MIX = 0.75 -- a 1-month clock the model does not use, and the revenue-growth
# gate that was RETIRED for ranking negatively. The public Vision site was
# therefore advertising a different model from the one running, with numbers from
# the pre-Sharadar survivorship-biased panel. Deriving from EDGE_SPEC means the
# export cannot drift from the product again (RESEARCH_RULES #7).
HOLD = edge_lib.EDGE_SPEC["hold"]
LONG_WINDOW = "20Y"   # full ~20-year track-record curve, shown alongside the primary
# The per-window summary stats ride along in the export via bt["windows"],
# regardless of which WINDOW drives the headline curve above.
# basket size n=10 is the product book (edge_lib.EDGE_SPEC / edge_tracker_lib.N)

def _rebal_label(spec: dict) -> str:
    """How the rebalance clock is described on the product page.

    Reads rebal_months, not hold. Since 2026-07-30 the grid is calendar-anchored
    (books are dated the first trading day of a month), so the honest label is a
    number of months; `hold` is only the trading-day approximation used for
    annualization and would drift from the real clock if the two disagreed."""
    m = spec.get("rebal_months")
    if m:
        return f"{int(m)}M"
    return f"{spec.get('hold_days', '?')}d"

# Fail loud at import time if either window label drifts from the model's list.
# A raise, not an assert: asserts are stripped under `python -O`, which would
# silently disable exactly the guard we want most on a production host.
for _w in (WINDOW, LONG_WINDOW):
    if _w not in edge_lib.WINDOWS:
        raise ValueError(f"export window {_w!r} is not in edge_lib.WINDOWS {edge_lib.WINDOWS}")


def build(window: str = WINDOW, hold: int = HOLD) -> dict:
    """Build Vision's data payload at the given window / rebalance clock.
    The 20-year track-record curve uses the same clock, window=20Y."""
    # clock_spec so the grid and the annualization cannot be set apart; a bare
    # {"hold": 42} here would run a monthly grid and label it a 2-month clock.
    clock = edge_lib.clock_spec(hold)
    bt = edge_lib.run_edge_backtest(window=window, spec=clock)
    if not bt.get("ok"):
        raise RuntimeError(f"backtest failed: {bt.get('reason')}")
    lt = edge_lib.run_edge_backtest(window=LONG_WINDOW, spec=clock)
    if not lt.get("ok"):
        raise RuntimeError(f"long backtest failed: {lt.get('reason')}")
    tr = edge_tracker_lib.tracker_state(hold=hold, window=window)
    if not tr.get("ok"):
        raise RuntimeError(f"tracker failed: {tr.get('reason')}")

    perf = bt["performance"]["model"]
    lperf = lt["performance"]["model"]
    spec = bt["spec"]
    # `signal`, not `accel`: the book key was renamed when acceleration was
    # retired, and reading the old name published a null for every holding.
    # `rows` keeps the tracker's full record (pctile / n_eligible / mcap), which
    # the write-up needs; `book` is the trimmed shape Vision publishes. Those
    # ranking internals stay out of the payload -- the page has no use for them
    # and they would be one more thing to keep consistent on the far side.
    rows = tr["current_book"]
    book = [{
        "ticker": b["ticker"], "name": b["name"], "sector": b["sector"],
        "weight": b["weight"], "signal": b.get("signal"),
        "signal_col": b.get("signal_col"), "price": b.get("price"),
    } for b in rows]

    # Industry + write-up per holding. `what` is fetched once per ticker and
    # cached forever; `standing` and `why` are recomputed each run because they
    # quote this month's numbers. A new name entering the book is described
    # automatically, with no list to maintain. Never fatal: describe() returns an
    # entry per holding regardless. The book is data; this is commentary, and it
    # cannot change a number.
    try:
        desc = company_desc.describe(rows, tr["book_date"], cfg=spec)
    except Exception as e:                                      # noqa: BLE001
        desc = {}
        print(f"  warning: descriptions unavailable ({type(e).__name__}: {e})")
    for b in book:
        d = desc.get(str(b["ticker"]).upper(), {})
        b["industry"] = d.get("industry") or ""
        b["what"] = d.get("what") or ""
        b["standing"] = d.get("standing") or ""
        b["why"] = d.get("why") or ""

    return {
        "product": "Vision",
        "config": {"window": window, "rebalance": _rebal_label(spec),
                   "hold_days": spec.get("hold_days"),
                   "rebal_months": spec.get("rebal_months"),
                   "rebalance_day": "first trading day of the month",
                   "n": len(book),
                   "signal": spec.get("signal"),
                   "mcap_floor_bn": spec.get("mcap_floor_bn"),
                   "sector_cap": spec.get("sector_cap"),
                   "continuous_regime": spec.get("continuous_regime"),
                   "vol_target": spec.get("vol_target"),
                   "cost_bps": spec.get("cost_bps", 10)},
        "as_of": tr["book_date"],
        "book_date": tr["book_date"],
        "book": book,
        "performance": {k: perf.get(k) for k in
                        ("cagr", "sharpe", "max_drawdown", "total_return")},
        "curve": {"dates": bt["curves"]["dates"],
                  "model": bt["curves"]["model"],
                  "sp500": bt["curves"]["sp500"]},
        "curve_long": {"window": LONG_WINDOW,
                       "dates": lt["curves"]["dates"],
                       "model": lt["curves"]["model"],
                       "sp500": lt["curves"]["sp500"]},
        "performance_long": {k: lperf.get(k) for k in
                             ("cagr", "sharpe", "max_drawdown", "total_return")},
        "windows": bt["windows"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": "Backtest, net of ~10bps costs. Not a forecast or investment advice.",
        # Say where each part came from. The business line is a third party's
        # text; the ranking and sector notes are our own arithmetic. A reader is
        # entitled to know which is which, and neither is a recommendation.
        "descriptions_note": ("Business descriptions are sourced from public company "
                              "profiles. The ranking and sector-cap notes are computed "
                              "from this model's own numbers. Neither is research, and "
                              "neither played any part in choosing the stocks."),
    }


def payload_js(data: dict) -> str:
    return ("// Auto-generated by export_vision.py — do not edit by hand.\n"
            "window.VISION_DATA = " + json.dumps(data, indent=2) + ";\n")


def write_local(payload: str, out: str | None = None) -> str:
    out = out or os.environ.get("VISION_LOCAL_OUT") or \
        os.path.join(os.path.dirname(__file__), "..", "vision", "vision_data.js")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(payload)
    return out


def push_to_github(payload: str, message: str) -> dict:
    """Commit vision_data.js to the Vision repo via the GitHub contents API.
    Needs GITHUB_TOKEN (contents:write). Repo/branch/path overridable via env."""
    # Two DIFFERENT credentials, deliberately. GITHUB_TOKEN is the deploy token:
    # it only needs Contents:READ on the private quantmodel repo so fetch_data can
    # pull the data bundle. Publishing needs Contents:WRITE on a DIFFERENT,
    # public repo. Fine-grained PATs are per-repository, so a token scoped to
    # quantmodel returns 404 here -- not 403 -- which reads like "no such repo"
    # rather than "no permission" and is why this failed silently-looking.
    # Prefer VISION_GITHUB_TOKEN so the deploy token never needs write anywhere.
    token = os.environ.get("VISION_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("VISION_REPO", "Tstansberry81/vision")
    branch = os.environ.get("VISION_BRANCH", "main")
    path = os.environ.get("VISION_DATA_PATH", "vision_data.js")
    if not token:
        return {"pushed": False, "reason": "no VISION_GITHUB_TOKEN / GITHUB_TOKEN"}
    import requests
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    sha = None
    g = requests.get(api, params={"ref": branch}, headers=headers, timeout=20)
    if g.status_code == 200:
        sha = g.json().get("sha")
    body = {"message": message, "branch": branch,
            "content": base64.b64encode(payload.encode("utf-8")).decode("ascii")}
    if sha:
        body["sha"] = sha
    p = requests.put(api, json=body, headers=headers, timeout=20)
    if p.status_code >= 300:
        hint = ""
        if p.status_code in (403, 404):
            which = "VISION_GITHUB_TOKEN" if os.environ.get("VISION_GITHUB_TOKEN") else "GITHUB_TOKEN"
            hint = (f" — {which} cannot write to {repo}. Fine-grained PATs are "
                    f"per-repository and return 404 (not 403) for repos outside "
                    f"their scope, so this looks like a missing repo. Give a token "
                    f"Contents:write on {repo} and set VISION_GITHUB_TOKEN.")
        return {"pushed": False, "reason": f"GitHub {p.status_code}{hint}"}
    return {"pushed": True, "commit": (p.json().get("commit") or {}).get("html_url"),
            "repo": repo, "branch": branch}


def export_to_vision(window: str = WINDOW, hold: int = HOLD,
                     actor: str | None = None) -> dict:
    """Build + publish. Pushes to the Vision repo if GITHUB_TOKEN is set (Render then
    auto-redeploys), else writes the local file (dev). JSON-friendly result."""
    data = build(window, hold)
    pj = payload_js(data)
    cfg = data["config"]
    msg = (f"Sync Vision — {cfg['window']} / {cfg['rebalance']} / "
           f"{cfg['n']} stocks / {cfg.get('signal', '?')}"
           + (f" (by {actor})" if actor else ""))
    if os.environ.get("GITHUB_TOKEN"):
        res = push_to_github(pj, msg)
    else:
        res = {"pushed": False, "local_path": write_local(pj), "reason": "no GITHUB_TOKEN — wrote local file"}
    return {
        "ok": bool(res.get("pushed") or res.get("local_path")),
        "pushed": bool(res.get("pushed")),
        "as_of": data["as_of"],
        "config": cfg,
        "book": [b["ticker"] for b in data["book"]],
        **{k: v for k, v in res.items() if k in ("commit", "local_path", "reason", "repo")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--hold", type=int, default=HOLD)
    args = ap.parse_args()

    data = build(args.window, args.hold)
    out = write_local(payload_js(data), args.out)
    print(f"Wrote {out}")
    print(f"  {data['config']['window']} / {data['config']['rebalance']} / "
          f"as of {data['as_of']} · "
          f"CAGR {data['performance']['cagr']*100:.1f}% / Sharpe {data['performance']['sharpe']:.2f}")
    print(f"  book: {', '.join(b['ticker'] for b in data['book'])}")


if __name__ == "__main__":
    main()
