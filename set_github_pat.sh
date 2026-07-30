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

# Verify WRITE, not just auth.
#
# This used to check `git ls-remote`, which only proves the token can READ. A
# fine-grained PAT with Contents:read passes that happily -- and then every push
# fails with 403 "Write access to repository not granted". That false pass is how
# a read-only token sat in this keychain undetected while five commits piled up
# locally. `push --dry-run` negotiates the real receive-pack permission and
# updates nothing, so it catches the difference.
#
# Both repos are checked. The keychain entry is per-HOST (github.com) so one
# token serves both, but a fine-grained PAT grants access per REPOSITORY and can
# easily cover one and not the other.
VISION_DIR="$(cd "$REPO_DIR/../vision" 2>/dev/null && pwd || true)"
echo "Stored. Verifying write access…"
rc=0
for d in "$REPO_DIR" ${VISION_DIR:+"$VISION_DIR"}; do
  git -C "$d" rev-parse --git-dir >/dev/null 2>&1 || continue
  name="$(basename "$d")"
  if err="$(git -C "$d" push --dry-run origin HEAD 2>&1)"; then
    ahead="$(git -C "$d" rev-list --count @{u}..HEAD 2>/dev/null || echo '?')"
    echo "  OK   $name — write confirmed, $ahead commit(s) ready to push."
  else
    rc=1
    reason="$(printf '%s' "$err" | grep -iE 'denied|not granted|403|401|not found' | head -1)"
    echo "  FAIL $name — ${reason:-see git output}" >&2
  fi
done
if [ "$rc" -ne 0 ]; then
  echo >&2
  echo "Stored, but it cannot WRITE everywhere it needs to." >&2
  echo "GitHub -> Settings -> Developer settings -> Personal access tokens ->" >&2
  echo "Fine-grained tokens -> your token:" >&2
  echo "  * Repository access must list BOTH quantmodel and vision" >&2
  echo "  * Repository permissions -> Contents -> Read and write" >&2
  exit 1
fi
echo
echo "All good. Push with:"
echo "  git -C \"$REPO_DIR\" push"
[ -n "$VISION_DIR" ] && echo "  git -C \"$VISION_DIR\" push"
