"""Render API helper — store the key safely, then read deploy/build state.

    python render_ops.py setup     # prompt for the key (hidden) and store it
    python render_ops.py status    # services + latest deploy state
    python render_ops.py deploys   # recent deploys, newest first
    python render_ops.py events    # recent service events (OOM, restarts, ...)
    python render_ops.py logs      # recent log lines
    python render_ops.py logs --grep fetch_data --limit 200

WHY A SCRIPT INSTEAD OF PASTING THE KEY
---------------------------------------
A secret pasted into a chat, a shell command, or a `--key=` argument leaks in
several places at once: the terminal scrollback, ~/.zsh_history, the process
table (`ps` shows every argument of every running command), and any transcript.
This script closes all of those:

  * the key is read with getpass -- not echoed, never in shell history
  * it is NEVER accepted as a command-line argument (see _reject_argv_key)
  * it is written with Python file I/O, so it never appears in argv or `ps`
  * it lands in .env, which is gitignored, and the file is chmod 600
  * nothing here ever prints the key; `setup` prints only a masked fingerprint

Get a key at: Render Dashboard -> Account Settings -> API Keys -> Create API Key.
It looks like `rnd_XXXXXXXXXXXXXXXXXXXXXXXX`. A Render API key is account-wide
and can modify services, so treat it like a password: if it ever appears in a
chat window or a log, rotate it in that same dashboard.
"""
from __future__ import annotations
import argparse
import getpass
import json
import os
import pathlib
import re
import sys

try:
    import requests
except ImportError:                                     # pragma: no cover
    sys.exit("render_ops: `pip install requests` first.")

import config                                            # loads .env for every entrypoint

API = "https://api.render.com/v1"
ENV_PATH = pathlib.Path(__file__).parent / ".env"
KEY_NAME = "RENDER_API_KEY"


def _reject_argv_key() -> None:
    """Refuse a key passed on the command line — argv is world-readable via `ps`."""
    for a in sys.argv[1:]:
        if a.startswith("rnd_") or "rnd_" in a:
            sys.exit("render_ops: do NOT pass the key as an argument — every argument is "
                     "visible in `ps` and saved to your shell history.\n"
                     "  Run `python render_ops.py setup` and paste it at the hidden prompt.\n"
                     "  That key is now exposed; rotate it in the Render dashboard.")


def _key() -> str:
    k = (os.environ.get(KEY_NAME) or "").strip()
    if not k:
        sys.exit(f"render_ops: no {KEY_NAME} found. Run `python render_ops.py setup` first.")
    return k


def _get(path: str, **params):
    r = requests.get(f"{API}{path}", headers={"Authorization": f"Bearer {_key()}",
                                              "Accept": "application/json"},
                     params=params, timeout=60)
    if r.status_code == 401:
        sys.exit("render_ops: 401 — the stored key is invalid or was revoked. Re-run setup.")
    if r.status_code != 200:
        sys.exit(f"render_ops: {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _mask(k: str) -> str:
    return f"{k[:8]}…{k[-4:]} ({len(k)} chars)" if len(k) > 14 else "(short key)"


def _write_env(key: str) -> None:
    """Upsert KEY_NAME in .env without reading the other secrets into memory
    any longer than needed, and without ever passing the value through argv."""
    lines, found = [], False
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            if re.match(rf"\s*{KEY_NAME}\s*=", ln):
                lines[i] = f"{KEY_NAME}={key}"
                found = True
    if not found:
        lines.append(f"{KEY_NAME}={key}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(ENV_PATH, 0o600)


def cmd_setup(_args) -> None:
    print("Paste your Render API key. It will NOT be shown as you type.")
    print("(Render Dashboard -> Account Settings -> API Keys -> Create API Key)\n")
    key = getpass.getpass("Render API key: ").strip()
    if not key:
        sys.exit("render_ops: nothing entered.")
    if not key.startswith("rnd_"):
        print("  warning: Render keys normally start with 'rnd_' — continuing anyway.")

    # Validate BEFORE storing, so a typo fails now instead of at 2am.
    r = requests.get(f"{API}/services", headers={"Authorization": f"Bearer {key}"},
                     params={"limit": 1}, timeout=60)
    if r.status_code == 401:
        sys.exit("render_ops: Render rejected that key (401). Nothing was saved.")
    if r.status_code != 200:
        sys.exit(f"render_ops: validation call failed ({r.status_code}): {r.text[:200]}. "
                 f"Nothing was saved.")

    _write_env(key)
    print(f"\n  stored {KEY_NAME} in {ENV_PATH.name} (chmod 600, gitignored)")
    print(f"  key    {_mask(key)}")
    print("  validated against the Render API — it works.")
    print("\nThe key was never echoed, never passed as an argument, and is not in your\n"
          "shell history. Claude can now read it from .env without you pasting it in chat.")


def _services():
    out = _get("/services", limit=50)
    # the API returns [{"service": {...}, "cursor": "..."}, ...]
    return [row.get("service", row) for row in out] if isinstance(out, list) else []


def _pick(svcs, want: str | None):
    if want:
        for s in svcs:
            if want in (s.get("id"), s.get("name")):
                return s
    for s in svcs:
        if "quant" in (s.get("name") or "").lower():
            return s
    return svcs[0] if svcs else None


def cmd_status(args) -> None:
    svcs = _services()
    if not svcs:
        sys.exit("render_ops: no services visible for this key.")
    for s in svcs:
        print(f"{s.get('name'):<24} {s.get('type','?'):<12} {s.get('id')}")
        su = s.get("serviceDetails", {}) or {}
        if su.get("url"):
            print(f"  url      {su['url']}")
        if su.get("numInstances") is not None:
            print(f"  plan     {su.get('plan','?')} · instances {su['numInstances']}")
    s = _pick(svcs, args.service)
    if not s:
        return
    print(f"\nlatest deploys for {s.get('name')}:")
    for row in _get(f"/services/{s['id']}/deploys", limit=5):
        d = row.get("deploy", row)
        print(f"  {d.get('status','?'):<12} {d.get('finishedAt') or d.get('createdAt','')}  "
              f"{(d.get('commit') or {}).get('message','')[:60]}")


def cmd_deploys(args) -> None:
    s = _pick(_services(), args.service)
    for row in _get(f"/services/{s['id']}/deploys", limit=args.limit):
        d = row.get("deploy", row)
        print(f"{d.get('status','?'):<12} {d.get('createdAt','')}  {d.get('id')}")
        c = d.get("commit") or {}
        if c:
            print(f"             {c.get('id','')[:8]} {c.get('message','')[:70]}")


def cmd_events(args) -> None:
    s = _pick(_services(), args.service)
    for row in _get(f"/services/{s['id']}/events", limit=args.limit):
        e = row.get("event", row)
        det = e.get("details") or {}
        print(f"{e.get('timestamp','')}  {e.get('type','?'):<26} "
              f"{json.dumps(det, default=str)[:120]}")


def cmd_logs(args) -> None:
    svcs = _services()
    s = _pick(svcs, args.service)
    owner = s.get("ownerId") or (s.get("owner") or {}).get("id")
    if not owner:
        sys.exit("render_ops: could not determine ownerId for the logs API.")
    out = _get("/logs", ownerId=owner, resource=s["id"], limit=args.limit)
    logs = out.get("logs", out) if isinstance(out, dict) else out
    pat = re.compile(args.grep) if args.grep else None
    for ln in logs:
        msg = ln.get("message", "") if isinstance(ln, dict) else str(ln)
        if pat and not pat.search(msg):
            continue
        print(f"{(ln.get('timestamp','') if isinstance(ln, dict) else '')[:19]}  {msg.rstrip()}")


def main() -> None:
    _reject_argv_key()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    for name, fn in (("status", cmd_status), ("deploys", cmd_deploys),
                     ("events", cmd_events), ("logs", cmd_logs)):
        p = sub.add_parser(name)
        p.add_argument("--service", help="service id or name (default: the quant one)")
        p.add_argument("--limit", type=int, default=20)
        if name == "logs":
            p.add_argument("--grep", help="only show lines matching this regex")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
