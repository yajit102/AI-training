#!/usr/bin/env bash
# Build a standalone downloadable .tar.gz of the FlowPredict pipeline —
# for handing to someone deploying without cloning the git repo directly.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="$(cat deploy/VERSION)"
OUT_DIR="dist"
PKG_NAME="flowpredict-${VERSION}"
STAGE_DIR="$(mktemp -d)/${PKG_NAME}"

echo "==> Staging package contents for v${VERSION}"
mkdir -p "${STAGE_DIR}"
cp -R src "${STAGE_DIR}/src"
cp -R scripts "${STAGE_DIR}/scripts"
cp -R tests "${STAGE_DIR}/tests"
cp -R docs "${STAGE_DIR}/docs"
cp -R deploy "${STAGE_DIR}/deploy"
cp requirements.txt "${STAGE_DIR}/requirements.txt"
cp README.md "${STAGE_DIR}/README.md" 2>/dev/null || true
find "${STAGE_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

mkdir -p "${OUT_DIR}"
tar -czf "${OUT_DIR}/${PKG_NAME}.tar.gz" -C "$(dirname "${STAGE_DIR}")" "${PKG_NAME}"
rm -rf "$(dirname "${STAGE_DIR}")"

echo "==> Built ${OUT_DIR}/${PKG_NAME}.tar.gz"
