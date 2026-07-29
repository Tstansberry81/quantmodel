# Deploying the live site (Render)

This is a Flask app with heavy compute (a ~179 MB backtest panel, 1–2 min runs,
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
- **Memory:** the panel peaks ~0.8–1.2 GB in RAM. `render.yaml` uses the **standard
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
