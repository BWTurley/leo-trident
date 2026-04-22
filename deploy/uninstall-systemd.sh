#!/usr/bin/env bash
# Stop, disable, and remove the leo-trident systemd user service.
# Idempotent: safe to re-run when the unit is already absent.
set -euo pipefail

UNIT_DST="${HOME}/.config/systemd/user/leo-trident.service"

if systemctl --user list-unit-files leo-trident.service >/dev/null 2>&1; then
  systemctl --user disable --now leo-trident.service 2>/dev/null || true
fi

systemctl --user stop leo-trident.service 2>/dev/null || true

if [ -e "${UNIT_DST}" ] || [ -L "${UNIT_DST}" ]; then
  rm -f "${UNIT_DST}"
fi

systemctl --user daemon-reload
systemctl --user reset-failed leo-trident.service 2>/dev/null || true

echo "OK: leo-trident user service removed."
