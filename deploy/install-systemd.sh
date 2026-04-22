#!/usr/bin/env bash
# Install and start the leo-trident systemd user service.
# Idempotent: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="${REPO_ROOT}/deploy/systemd/leo-trident.service"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/leo-trident.service"
LOG_DIR="${REPO_ROOT}/logs"
HEALTH_URL="http://127.0.0.1:8765/health"

if [ ! -f "${UNIT_SRC}" ]; then
  echo "ERROR: unit file not found at ${UNIT_SRC}" >&2
  exit 1
fi

if [ ! -f "${REPO_ROOT}/.env" ]; then
  echo "ERROR: ${REPO_ROOT}/.env is missing. Copy deploy/env.template and fill in secrets." >&2
  exit 1
fi

mkdir -p "${UNIT_DIR}"
mkdir -p "${LOG_DIR}"

# Replace any existing symlink/file with a fresh symlink to the repo unit.
ln -sfn "${UNIT_SRC}" "${UNIT_DST}"

systemctl --user daemon-reload
systemctl --user enable --now leo-trident.service

# Wait up to 10 seconds for /health to return 200.
deadline=$(( $(date +%s) + 10 ))
http_code=000
while [ "$(date +%s)" -lt "${deadline}" ]; do
  http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "${HEALTH_URL}" 2>/dev/null) || http_code=000
  if [ "${http_code}" = "200" ]; then
    break
  fi
  sleep 1
done

systemctl --user status leo-trident --no-pager || true

if [ "${http_code}" != "200" ]; then
  echo "ERROR: ${HEALTH_URL} did not return 200 within 10s (last code: ${http_code})" >&2
  exit 1
fi

echo "OK: leo-trident is running and ${HEALTH_URL} returned 200."

# Self-verify the install by running the full smoke harness.
SMOKE_SH="${REPO_ROOT}/scripts/smoke.sh"
if [ -x "${SMOKE_SH}" ]; then
  echo "Running smoke harness..."
  if ! bash "${SMOKE_SH}"; then
    echo "ERROR: smoke harness failed — install is NOT verified." >&2
    exit 1
  fi
else
  echo "WARN: ${SMOKE_SH} not found or not executable — skipping post-install smoke." >&2
fi
