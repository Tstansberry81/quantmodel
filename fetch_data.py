"""Deploy-time data fetcher (runs in the Render build step).

The data artifacts (backtest_data.pkl etc.) are too big to commit to git, so on a
fresh host we download a prebuilt bundle from the DATA_URL env var and extract it.
If the artifacts are already present (e.g. local dev), this is a no-op.

Create the bundle locally with `python make_data_bundle.py`, upload data_bundle.zip
somewhere with a direct-download link, and set DATA_URL to it in the Render
dashboard.

PRIVATE REPO / LICENSED DATA
----------------------------
The artifacts are Sharadar-derived and Sharadar is licensed per seat, so the
bundle must NOT sit anywhere public. This repo is private, which means its
Release assets are private too -- and a private asset returns 404 to an
unauthenticated request. Set GITHUB_TOKEN in the Render environment (a
fine-grained PAT with Contents:read on this repo) and this fetcher authenticates.

GitHub's REST download for a release asset also requires
`Accept: application/octet-stream`; without it the API returns the asset's JSON
metadata, which then fails to unzip with a confusing error.
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

import config

art = config.ARTIFACT_DIR / "backtest_data.pkl"
# Version marker. "artifacts already present -> skip" is right for local dev but
# silently wrong the moment the BUNDLE changes: on any host that preserves the
# directory between deploys, a stale panel survives forever and the site keeps
# serving old data while every deploy reports success.
marker = config.ARTIFACT_DIR / ".data_version"
have = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""

url = os.environ.get("DATA_URL", "").strip()
headers = {"User-Agent": "quant-model-deploy"}
token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("DATA_TOKEN") or "").strip()
if token:
    headers["Authorization"] = f"Bearer {token}"


def _resolve_release_asset(tag_url: str) -> tuple[str, str]:
    """Point DATA_URL at a release TAG and let this find the asset under it.

    WHY: a GitHub release asset gets a NEW numeric id on every upload, so a
    DATA_URL pinned to an asset id goes stale the moment the bundle is
    republished. That is what forced TWO environment variables to be hand-edited
    on every data change -- DATA_URL because the id moved, and DATA_VERSION to
    defeat the "already present -> skip" path. A tag URL is stable forever: the
    asset is located by NAME underneath it, and its id+updated_at becomes the
    version string, so neither variable has to be touched again.

    Returns (download_url, auto_version).
    """
    req = urllib.request.Request(
        tag_url, headers={**headers, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as r:
        rel = json.loads(r.read().decode("utf-8"))
    name = os.environ.get("DATA_ASSET_NAME", "data_bundle.zip").strip()
    assets = rel.get("assets") or []
    match = next((a for a in assets if a.get("name") == name), None)
    if match is None:
        names = ", ".join(a.get("name", "?") for a in assets) or "(none)"
        raise SystemExit(
            f"[fetch_data] ERROR: release '{rel.get('tag_name')}' has no asset named "
            f"'{name}'. Assets present: {names}.\n  -> publish with "
            f"`python publish_bundle.py`, which always uses the fixed name.")
    # The id changes when a new asset is uploaded; updated_at changes if the same
    # asset is replaced. Either moving means the bytes changed -> re-download.
    return match["url"], f'{match["id"]}:{match.get("updated_at", "")}'


auto_version = ""
if url and "/releases/tags/" in url:
    # Resolving the tag needs the network, and it happens before we know whether
    # anything must be downloaded. So a failure here must NOT fail the build when
    # usable artifacts already exist: a DNS blip, a rate limit or a truncated
    # body would otherwise take down a deploy that needed no data at all. Catch
    # broadly (URLError, JSONDecodeError, KeyError -- none are HTTPError).
    try:
        url, auto_version = _resolve_release_asset(url)
        print(f"[fetch_data] resolved release tag -> asset {auto_version.split(':')[0]}")
    except SystemExit:
        raise                                  # our own "no such asset" message
    except Exception as e:
        code = getattr(e, "code", None)
        hint = (" (a private repo's releases need GITHUB_TOKEN with Contents:read)"
                if code in (401, 403, 404) else "")
        if art.exists():
            print(f"[fetch_data] WARNING: could not resolve the release tag{hint}: "
                  f"{type(e).__name__}: {e}\n  -> keeping the artifacts already on disk.")
            sys.exit(0)
        print(f"[fetch_data] ERROR: could not resolve the release tag and there are no "
              f"local artifacts to fall back on{hint}: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)

# With a tag URL the asset itself is authoritative. A DATA_VERSION left over in
# the dashboard from the old scheme must NOT win, or it pins `want` to a frozen
# string that already equals the marker and every future republish is skipped --
# silently reinstating the stale-bundle bug this scheme exists to remove.
manual = os.environ.get("DATA_VERSION", "").strip()
if manual and auto_version:
    print(f"[fetch_data] NOTE: ignoring leftover DATA_VERSION='{manual}' — the release "
          f"tag supplies the version. You can delete that variable.")
want = auto_version or manual

if art.exists() and (not want or want == have):
    why = ("no DATA_VERSION set and DATA_URL is not a release tag" if not want
           else f"version {have} already installed")
    print(f"[fetch_data] artifacts already present ({why}); nothing to download.")
    sys.exit(0)
if art.exists():
    print(f"[fetch_data] artifacts present but version '{have or 'unknown'}' != "
          f"'{want}' — re-downloading.")

if not url:
    print("[fetch_data] WARNING: artifacts missing and DATA_URL not set — the site "
          "will start but backtests/portfolio will be unavailable until data is provided.")
    sys.exit(0)

# Gate on the URL, NOT on the token: a resolved tag always produces an
# api.github.com asset URL, and without this header GitHub returns the asset's
# JSON metadata, which then dies in zipfile with a thoroughly confusing error.
if "api.github.com" in url:
    headers["Accept"] = "application/octet-stream"

# GUARD, checked BEFORE spending the download. The published bundle is always
# DEPLOY-TRIMMED (make_data_bundle keeps only names that ever cleared the model's
# market-cap floor), so extracting it over a developer's FULL research artifact
# destroys the wider universe that survivorship and small-cap work depend on.
# Learned the hard way: an end-to-end test of this fetcher, pointed at the real
# data dir, replaced a 12,164-name artifact with the 1,814-name deploy copy.
# Keyed on the POSITIVE marker the trimmed bundle writes, not a name count -- a
# meta.json without n_names scored 0 and let an earlier version fail OPEN.
# On a host the artifact always came FROM a bundle and carries the marker, so
# this only ever fires locally, which is the only place it can do damage.
local_meta = config.ARTIFACT_DIR / "meta.json"
if art.exists() and local_meta.exists() and not os.environ.get("FORCE_DATA_OVERWRITE"):
    try:
        _local = json.loads(local_meta.read_text(encoding="utf-8"))
    except Exception:
        _local = {}
    if _local.get("deploy_trim_mcap") is None:
        sys.exit(
            f"[fetch_data] REFUSING to overwrite: the local artifact "
            f"({_local.get('n_names', '?')} names) is NOT deploy-trimmed, and the "
            f"published bundle is. Extracting would destroy the wider research "
            f"universe.\n  -> set FORCE_DATA_OVERWRITE=1 if that is genuinely what "
            f"you want, or unset DATA_URL for local work.")

print("[fetch_data] downloading data bundle from DATA_URL ...")
req = urllib.request.Request(url, headers=headers)
tmp = None
try:
    # Stream to disk rather than into memory: the bundle is ~1GB and the Render
    # instance has 2GB total, so buffering it whole invites the OOM killer to
    # take out the build.
    with urllib.request.urlopen(req) as r, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as fh:
        tmp = fh.name
        shutil.copyfileobj(r, fh, length=1 << 22)
    size = os.path.getsize(tmp)
    print(f"[fetch_data] downloaded {size/1e6:.0f} MB; extracting to {config.ROOT} ...")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(config.ROOT)    # bundle holds data/artifacts/... and data/cache/...
except urllib.error.HTTPError as e:
    hint = ""
    if e.code in (401, 403, 404):
        hint = ("\n  -> this repo is PRIVATE, so its Release assets need auth. Set "
                "GITHUB_TOKEN (Contents:read) in the Render environment, and point "
                "DATA_URL at the api.github.com asset URL rather than the browser one.")
    print(f"[fetch_data] ERROR {e.code} fetching the bundle.{hint}", file=sys.stderr)
    sys.exit(1)
finally:
    if tmp and os.path.exists(tmp):
        os.remove(tmp)

if art.exists() and want:
    marker.write_text(want, encoding="utf-8")   # incl. the auto version from a tag URL

ok = art.exists()
if ok:
    try:
        import json as _json
        _m = _json.loads((config.ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))
        print(f"[fetch_data] done. OK — source={_m.get('source')} "
              f"names={_m.get('n_names')} built={_m.get('built_at')}")
    except Exception:
        print("[fetch_data] done. OK")
else:
    print("[fetch_data] WARNING: backtest_data.pkl still missing — check the bundle layout.")
