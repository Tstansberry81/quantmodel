# Deploying the live site (Render)

This is a Flask app with heavy compute (a ~141 MB backtest panel, 1–2 min runs,
pandas/scipy/sklearn/hmmlearn). It needs a **real Python web service**, not a
static/serverless host — Render (or Railway/Fly) works; **Netlify does not**.

Everything is pre-wired: `render.yaml` (blueprint), `Procfile` (gunicorn, 600s
timeout), `runtime.txt` (Python 3.12), `requirements.txt` (+gunicorn),
`fetch_data.py` (downloads the data bundle at build), `make_data_bundle.py`.

## ⚠️ Do this first — rotate the API key
The fiscal.ai key was hard-coded in `config.py` and is in git history (and the
zip we made earlier). **Rotate it at fiscal.ai.** The live site does NOT need it
(it serves from prebuilt artifacts); only `build_data.py` does, via the
`FISCAL_API_KEY` env var.

## Steps

1. **Push the repo to GitHub** (the code only — `data/` stays gitignored).

2. **Build + publish the data bundle** (artifacts are too big for git):
   ```
   ALLOW_LICENSED_BUNDLE=1 .venv-mac/bin/python make_data_bundle.py   # -> data_bundle.zip (~300 MB)
   .venv-mac/bin/python publish_bundle.py                             # -> the GitHub Release
   ```
   `publish_bundle.py` always uploads under the **fixed name** `data_bundle.zip`
   (staging upload → retire the old one → rename), so the release tag URL stays
   valid forever and there is never a window where the asset is missing.

   **No environment edits are needed on a republish.** `DATA_URL` points at the
   release *tag*, and `fetch_data.py` resolves the asset underneath it and derives
   the version from the asset's own `id:updated_at`. Pinning an asset *id* was
   what previously forced both `DATA_URL` and `DATA_VERSION` to be hand-edited
   every time — and forgetting either silently kept serving the old bundle.

3. **Create the Render service**: Dashboard → New → **Blueprint** → pick the repo
   (it reads `render.yaml`). Then set env vars:
   - `DATA_URL` = the release **TAG** API URL, e.g.
     `https://api.github.com/repos/<owner>/<repo>/releases/tags/data-v2-sharadar`
     (not an individual asset URL — those change id on every upload).
   - `GITHUB_TOKEN` = fine-grained PAT with **Contents: read** — the repo is
     private, so its release assets 404 without auth.
   - **Do NOT set `DATA_VERSION`.** It is a manual override for non-GitHub
     hosting; left over from the old scheme it would pin the version and skip
     every future republish. `fetch_data.py` warns if it finds one.
   - (`FISCAL_API_KEY` only if you'll rebuild data on the host — usually skip.)

4. **Deploy.** Build runs `fetch_data.py` (downloads + extracts the bundle); start
   runs gunicorn. The bundle ships a **prebuilt panel**, so the first backtest is
   ~2s rather than a rebuild — but note the panel fingerprint hashes `edge_lib.py`,
   so any code change to that file invalidates it and the first cold start after
   such a push pays one rebuild (cached to the `/var/data` disk thereafter).
   Republishing the bundle avoids even that. Previously — after that
   it's cached and fast.

## Notes / gotchas
- **Memory:** the shipped panel is 141 MB on disk; BUILDING one peaks far
  higher (it holds every name's price arrays at once). Two concurrent builds
  OOM-killed the 2 GB instance on 2026-07-30 — `load_edge_panel` now serializes
  builds, and a republished bundle means the host unpickles instead of
  building. `render.yaml` uses the **standard
  (2 GB)** plan. The free/starter 512 MB tiers will likely OOM.
- **Cold starts:** on plans that sleep when idle, the first visitor after a sleep
  waits for spin-up + the first backtest. A non-sleeping plan avoids this.
- **Abuse/cost:** the backtest is heavy and public — anyone can trigger it.
  Consider a simple rate-limit or basic-auth if traffic is a concern.
- **Honesty caveats** (survivorship, momentum-regime, lumpy short-window CAGRs)
  are already shown on the Edge page — keep them; don't quote short-window returns
  as expected performance.
- **Railway/Fly** work the same way (Procfile + `DATA_URL`); only the dashboard
  differs.

## Concurrency and memory (learned the hard way, 2026-07-30)

Four OOM kills in one afternoon on the 2 GB plan, four distinct causes. Worth
reading before changing anything in the request path.

**The shape of the problem.** One worker, four threads, and several operations
that each hold hundreds of MB while running. Nothing was bounded, so any two of
them overlapping could exceed the ceiling. Each fix below was real and none was
sufficient alone — which is why the site 502'd repeatedly while they landed.

1. **Concurrent panel builds.** `lru_cache` takes no lock across the wrapped
   call, so the boot warm-up thread and an incoming request both missed and both
   built the panel. A build peaks far above the finished 141 MB pickle because it
   holds every name's price arrays at once. → build lock in `load_edge_panel`.

2. **An oversized response cache.** `_cached_tracker` was `maxsize=1024` at
   489 KB per payload — ~500 MB of ceiling. The comment justifying it ("~tens of
   KB → ~25MB") was written for `_cached_edge_backtest`, whose payload really is
   12 KB, and was never re-checked against this endpoint. The monthly clock then
   doubled the payload. → `maxsize=12`.

3. **Concurrent computes.** Nothing between the request handler and the panel
   took a lock, so four cold requests each built their own daily series.
   → one process-wide compute lock.

4. **Queued requests recomputing instead of sharing.** `lru_cache` does not
   dedupe IN-FLIGHT calls — it stores a result only when the call returns. So
   everything waiting on the lock recomputed the same thing and the queue never
   drained. → `_computed(key, fn)` re-checks a memo after acquiring.

**Rules that follow.**

- A cache size is a memory budget. Measure the payload before choosing one, and
  never copy a size across endpoints — measure that endpoint's payload too.
- Any lock in the request path must be BOUNDED. An unbounded wait starved
  `/api/meta`, which is the `healthCheckPath`, and Render restarts an instance
  whose health check stops answering. Trading an OOM for that is not a fix.
- Anything guarded by a lock needs a double-check after acquiring, or waiters
  duplicate the work they queued for.
- Concurrency was never buying throughput here (one worker, CPU-bound work).
  The caches are what make it fast. Serializing costs nothing real.

**Never edit `edge_lib.py` after publishing a bundle.** The panel fingerprint
hashes that file, so the shipped panel is instantly stale and every cold start
rebuilds it — silently, because a stale panel is a cache miss and not an error.
`publish_bundle.py` now refuses to ship a bundle whose panel filename doesn't
match the current spec, or whose `edge_lib.py` has uncommitted changes.

## The monthly rebalance runs itself (2026-09-01)

The model's clock is the first **trading** day of every month. Until now nothing
ran it: the data bundle was refreshed by hand, so after 2026-07-30 the panel
froze, the **2026-08-03 rebalance never happened**, and the site kept presenting
the 2026-07-01 book as current for a full month. `.github/workflows/monthly-rebalance.yml`
is the fix.

### What it does

1. `sharadar_ingest.py all` — pull the bulk tables (~1.7GB), rebuild the artifact
2. `make_data_bundle.py` — trim, **precompute the panel**, zip (~340MB)
3. `publish_bundle.py` — replace `data_bundle.zip` on the `data-v2-sharadar` release
4. `scripts/post_deploy.py` — trigger the Render deploy, **wait for `/api/meta`'s
   `built_at` to move**, then `GET /api/edge_tracker` (this is what writes the
   forward-record snapshot) and `POST /api/sync_vision`

Steps 1–3 need credentials; step 4 needs none — the deploy hook is an unguessable
URL and both app endpoints are unauthenticated.

### Secrets to set

Repo → Settings → Secrets and variables → Actions:

| Secret | Where to get it |
| --- | --- |
| `NASDAQ_DATA_LINK_API_KEY` | your Sharadar seat key |
| `RENDER_DEPLOY_HOOK` | Render → quantmodel → Settings → Deploy Hook |

The release upload uses the built-in `GITHUB_TOKEN` (`permissions: contents: write`).

### Why a gate instead of a cron expression

Cron cannot say "first trading day" — the 1st is a weekend or holiday about a
third of the time. So the workflow wakes every weekday of the first eight days at
21:30 UTC and `scripts/rebalance_gate.py` decides.

It waits for the **entry bar**, not the book date. `edge_lib` prices entry at the
t+1 close, so the panel cannot produce a live book for rebalance date `d` until
the session *after* `d` has closed. Running earlier yields an empty live panel.

### Known residual: a Friday book date

`_logged_late` marks a book written down more than **3 calendar days** after its
own date as `BACKFILLED` — kept for provenance, never scored. The job normally
fires on the entry bar's evening, a 1-day lag. But when a month's first trading
day is a **Friday**, the entry bar is the following Monday and the earliest
possible lag is already 3. If Sharadar has not published Monday's close by
21:30 UTC, the retry lands Tuesday at lag 4 and the row is `BACKFILLED`.

The gate prints a `::warning::` when this happens rather than letting it pass
silently. The real fix is to measure lateness in **trading sessions** rather than
calendar days — logging after the entry bar's close but before the next close
carries zero hindsight regardless of how many weekend days intervene — but that
changes an integrity guard, so it is left as a deliberate follow-up.

### Running it by hand

Actions → monthly-rebalance → Run workflow. `force` skips the gate; `skip_deploy`
builds and publishes the bundle without touching production.
