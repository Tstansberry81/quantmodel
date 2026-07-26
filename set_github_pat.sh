#!/usr/bin/env bash
# Store a GitHub PAT in the macOS keychain for this repo's pushes.
#
# The token is read from a silent prompt, so it never appears in your shell
# history, in the process list, or on screen. It goes straight into the
# keychain via git's own credential helper -- NOT into .git/config, which
# would leave it sitting in plaintext inside the repo.
#
#   ./set_github_pat.sh
#
# Then just `git push` as normal.
set -euo pipefail

USER_NAME="${1:-Tstansberry81}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf 'Paste GitHub PAT for %s (input hidden), then press Enter: ' "$USER_NAME"
IFS= read -rs PAT
printf '\n'

if [ -z "${PAT}" ]; then
  echo "No token entered — nothing changed." >&2
  exit 1
fi
case "$PAT" in
  github_pat_*|ghp_*|gho_*) ;;
  *) echo "That doesn't look like a GitHub token (expected github_pat_… or ghp_…)." >&2
     unset PAT; exit 1 ;;
esac

# Drop any stale credential first, otherwise the old one can keep winning.
printf 'protocol=https\nhost=github.com\n\n' | git -C "$REPO_DIR" credential reject 2>/dev/null || true
printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n\n' "$USER_NAME" "$PAT" \
  | git -C "$REPO_DIR" credential approve
unset PAT

echo "Stored. Verifying access…"
if git -C "$REPO_DIR" ls-remote --exit-code origin HEAD >/dev/null 2>&1; then
  ahead="$(git -C "$REPO_DIR" rev-list --count origin/main..HEAD 2>/dev/null || echo '?')"
  echo "OK — authenticated. $ahead local commit(s) ready to push."
  echo "Run:  git -C \"$REPO_DIR\" push origin main"
else
  echo "FAILED — the token was stored but GitHub rejected it." >&2
  echo "Check it hasn't expired and that it grants Contents:read/write on" >&2
  echo "Tstansberry81/quantmodel and Tstansberry81/vision." >&2
  exit 1
fi
