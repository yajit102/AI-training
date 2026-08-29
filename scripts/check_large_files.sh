#!/usr/bin/env bash
# Fails if a staged (or, with --all, tracked) file over $MAX_BYTES is about to
# be committed as a regular git blob instead of via Git LFS.
set -euo pipefail

MAX_BYTES=$((5 * 1024 * 1024)) # 5MB

if [[ "${1:-}" == "--all" ]]; then
  files=$(git ls-files)
else
  files=$(git diff --cached --name-only --diff-filter=ACM)
fi

lfs_files=$(git lfs ls-files -n 2>/dev/null || true)

status=0
for f in $files; do
  [[ -f "$f" ]] || continue
  if grep -qxF "$f" <<<"$lfs_files"; then
    continue # already tracked by LFS
  fi
  size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
  if (( size > MAX_BYTES )); then
    echo "ERROR: '$f' is $((size / 1024 / 1024))MB and not tracked by Git LFS." >&2
    echo "       Either 'git lfs track' its extension, or (if it's raw data" >&2
    echo "       or a full checkpoint) do NOT commit it — write a manifest" >&2
    echo "       instead. See docs/DATA_MANAGEMENT.md." >&2
    status=1
  fi
done

exit $status
