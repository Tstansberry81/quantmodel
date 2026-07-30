"""Publish data_bundle.zip to the GitHub release under a FIXED asset name.

    python make_data_bundle.py      # build it  (needs ALLOW_LICENSED_BUNDLE=1)
    python publish_bundle.py        # ship it   (no environment edits, ever)

WHY THIS EXISTS. Every upload used to create a new release asset with a new
numeric id, so DATA_URL had to be re-pointed by hand -- and because the fetcher
skips when artifacts are already present, DATA_VERSION had to be bumped by hand
too. Two manual environment edits per data change, each of which silently serves
stale numbers if forgotten.

This uploads under one constant name (data_bundle.zip), deleting any existing
asset of that name first, so DATA_URL can point at the release TAG forever and
fetch_data.py resolves the current asset underneath it. The version is derived
from the asset's id + updated_at, so a republish is detected automatically.

Credentials are read from the git credential helper (macOS keychain) or
GITHUB_TOKEN, and are never echoed or written to disk.
"""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import sys

try:
    import requests
except ImportError:                                     # pragma: no cover
    sys.exit("publish_bundle: `pip install requests` first.")

REPO = os.environ.get("DATA_REPO", "Tstansberry81/quantmodel")
TAG = os.environ.get("DATA_TAG", "data-v2-sharadar")
ASSET = os.environ.get("DATA_ASSET_NAME", "data_bundle.zip")
BUNDLE = pathlib.Path(__file__).parent / "data_bundle.zip"


def _token() -> str:
    """Env first, then the git credential helper. Never printed."""
    tok = (os.environ.get("GITHUB_TOKEN") or os.environ.get("DATA_TOKEN") or "").strip()
    if tok:
        return tok
    try:
        out = subprocess.run(["git", "credential", "fill"],
                             input="protocol=https\nhost=github.com\n\n",
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    sys.exit("publish_bundle: no credential found (set GITHUB_TOKEN or run `git credential fill`).")


def _check_panel_matches_code() -> None:
    """Refuse to publish a bundle whose panel the deployed code will reject.

    The panel fingerprint hashes edge_lib.py. Change one comment in that file
    after building the bundle and the shipped panel is dead on arrival: every
    cold start rebuilds it, which takes minutes, and on a small instance those
    rebuilds are exactly what makes the site look hung.

    Lived through on 2026-07-30 -- the bundle was published and edge_lib.py was
    edited eleven minutes later, so the republish bought nothing and every
    request timed out behind a rebuild. The failure is silent by construction
    (a stale panel is a cache miss, not an error), which is why this is a
    pre-flight check rather than something to notice afterwards.
    """
    import zipfile
    import edge_lib as E
    want_hold = E.EDGE_SPEC["hold"]
    want = E._panel_cache_paths(**{**E.clock_spec(want_hold),
                                   "universe": E.UNIVERSE, "offset_days": 0})[1].name
    with zipfile.ZipFile(BUNDLE) as z:
        names = [n.rsplit("/", 1)[-1] for n in z.namelist()]
    if want not in names:
        shipped = [n for n in names if n.startswith("panel_")] or ["(none)"]
        sys.exit(f"publish_bundle: REFUSING — the bundle carries {shipped} but the\n"
                 f"  current code asks for {want}. Rebuild with make_data_bundle.py.")

    # Same file, same bytes: the fingerprint hashes edge_lib.py itself, so an
    # uncommitted edit means the panel matches your working tree and not what
    # will be deployed.
    dirty = subprocess.run(["git", "status", "--porcelain", "edge_lib.py"],
                           capture_output=True, text=True,
                           cwd=BUNDLE.parent).stdout.strip()
    if dirty:
        sys.exit("publish_bundle: REFUSING — edge_lib.py has uncommitted changes.\n"
                 "  The bundled panel is fingerprinted against your working tree; the\n"
                 "  host will run the committed file and reject it. Commit, rebuild,\n"
                 "  then publish.")
    # THE BUILD KNOBS, not just the filename. The filename encodes hold /
    # universe / offset / rebal_months -- it does NOT encode USE_PIT_UNIVERSE or
    # DELIST_HAIRCUT, which _panel_fingerprint also hashes. A panel built on a
    # machine that auto-detects a PIT source carries the right NAME and the wrong
    # KEY, so the host rejects it silently and rebuilds on EVERY cold start. That
    # went unnoticed for months and was the real reason production was slow.
    #
    # The fingerprint itself cannot be recomputed here: it hashes the artifact
    # size and meta.json, and make_data_bundle swaps the deploy-trimmed copies
    # back out after building. So the build records its knobs and this checks
    # those against what render.yaml pins.
    import json as _js
    with zipfile.ZipFile(BUNDLE) as z:
        manifest = next((n for n in z.namelist() if n.endswith("panel_build.json")), None)
        if manifest is None:
            sys.exit("publish_bundle: REFUSING — no panel_build.json in the bundle.\n"
                     "  Rebuild with make_data_bundle.py (it records the build knobs).")
        knobs = _js.loads(z.read(manifest))
    if knobs.get("use_pit_universe") is not False:
        sys.exit(f"publish_bundle: REFUSING — the panel was built with "
                 f"USE_PIT_UNIVERSE={knobs.get('use_pit_universe')}, but production "
                 f"pins EDGE_USE_PIT_UNIVERSE=0.\n"
                 f"  The host would reject this panel and rebuild on every cold "
                 f"start — and it is a different universe besides.")
    if knobs.get("hold") != want_hold:
        sys.exit(f"publish_bundle: REFUSING — panel built for hold={knobs.get('hold')}, "
                 f"spec says {want_hold}.")
    print(f"  pre-flight OK — {want}, built with USE_PIT_UNIVERSE=False, "
          f"edge_lib.py committed")


def main() -> None:
    if not BUNDLE.exists():
        sys.exit(f"publish_bundle: {BUNDLE} not found — run make_data_bundle.py first.")
    _check_panel_matches_code()
    tok = _token()
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    base = f"https://api.github.com/repos/{REPO}"

    r = requests.get(f"{base}/releases/tags/{TAG}", headers=h, timeout=60)
    if r.status_code != 200:
        sys.exit(f"publish_bundle: release '{TAG}' lookup failed ({r.status_code}): "
                 f"{r.json().get('message', '')}")
    rel = r.json()

    # UPLOAD FIRST, THEN SWAP. Deleting the live asset before uploading its
    # replacement opens a window where the release has no asset of this name at
    # all -- and because DATA_URL now resolves by name instead of pinning an id,
    # any deploy landing in that window fails outright with "release has no asset
    # named ...". A 300MB upload over a flaky link is exactly when that happens.
    # GitHub rejects a duplicate name, so upload under a staging name, delete the
    # old one only once the bytes are safely up, then PATCH the name into place.
    staging = f"{ASSET}.incoming"
    for a in rel.get("assets", []):
        if a.get("name") == staging:            # leftover from an earlier failure
            requests.delete(f"{base}/releases/assets/{a['id']}", headers=h, timeout=60)

    size_mb = BUNDLE.stat().st_size / 1e6
    print(f"  uploading {size_mb:.0f} MB as {staging} ...")
    with open(BUNDLE, "rb") as fh:
        up = requests.post(rel["upload_url"].split("{")[0], params={"name": staging},
                           headers={**h, "Content-Type": "application/zip"},
                           data=fh, timeout=3600)
    if up.status_code not in (200, 201):
        sys.exit(f"publish_bundle: upload failed ({up.status_code}): {up.text[:300]}\n"
                 f"  the previous {ASSET} is untouched and still serving.")
    new_id = up.json()["id"]

    for a in rel.get("assets", []):
        if a.get("name") == ASSET:
            d = requests.delete(f"{base}/releases/assets/{a['id']}", headers=h, timeout=60)
            print(f"  retired previous asset {a['id']} ({d.status_code})")

    ren = requests.patch(f"{base}/releases/assets/{new_id}", headers=h,
                         json={"name": ASSET}, timeout=60)
    if ren.status_code != 200:
        sys.exit(f"publish_bundle: uploaded OK but rename failed ({ren.status_code}). "
                 f"Asset {new_id} is on the release as '{staging}' — rename it to "
                 f"'{ASSET}' by hand, or re-run.")
    a = ren.json()
    print(f"  done: asset {a['id']} · {a.get('state')} · {a['size']/1e6:.0f} MB")
    print()
    print("Nothing to change in the Render environment. It should already hold:")
    print(f"  DATA_URL = https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    print("  (no DATA_VERSION needed — the fetcher versions off the asset itself)")


if __name__ == "__main__":
    main()
