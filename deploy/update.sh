#!/usr/bin/env bash
# Pull the latest FlowPredict updates and redeploy the Docker image.
#
# Usage:
#   deploy/update.sh                 # pulls current branch, rebuilds, runs
#   FLOWPREDICT_BRANCH=main deploy/update.sh
#   FLOWPREDICT_ARGS="--n-cases 20000" deploy/update.sh
#
# Safe to re-run: it only fast-forwards the working tree (fails loudly on
# local uncommitted changes or diverged history rather than discarding
# anything) before rebuilding and re-running the container.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BRANCH="${FLOWPREDICT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
ARGS="${FLOWPREDICT_ARGS:---n-cases 8000}"

echo "==> Checking working tree is clean"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: uncommitted local changes present. Commit, stash, or discard them first." >&2
  exit 1
fi

echo "==> Fetching origin/${BRANCH}"
git fetch origin "${BRANCH}"

echo "==> Fast-forwarding local ${BRANCH} to origin/${BRANCH}"
git checkout "${BRANCH}"
if ! git merge --ff-only "origin/${BRANCH}"; then
  echo "ERROR: local branch has diverged from origin/${BRANCH}; resolve manually (do not force)." >&2
  exit 1
fi

echo "==> Rebuilding Docker image"
docker compose -f deploy/docker-compose.yml build

echo "==> Running updated container (args: ${ARGS})"
# shellcheck disable=SC2086
docker compose -f deploy/docker-compose.yml run --rm flowpredict ${ARGS}

echo "==> Done. Image 'flowpredict:latest' now reflects origin/${BRANCH}."
