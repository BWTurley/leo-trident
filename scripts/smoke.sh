#!/usr/bin/env bash
# Leo Trident end-to-end smoke harness.
#
# Exercises the full Hermes + Trident stack and reports green/red.
# Designed to complete in under 30 seconds. Exit 0 on all-pass, 1 on first
# failure. Each step prints [PASS] or [FAIL] with a one-line reason.
#
# Usage:
#   bash scripts/smoke.sh                 # against http://127.0.0.1:8765
#   TRIDENT_URL=http://1.2.3.4:9000 bash scripts/smoke.sh
#   SKIP_SYSTEMD=1 bash scripts/smoke.sh  # skip systemd active check
set -uo pipefail

TRIDENT_URL="${TRIDENT_URL:-http://127.0.0.1:8765}"
TRIDENT_HOME="${LEO_TRIDENT_HOME:-${HOME}/leo_trident}"
DB_PATH="${LEO_DB_PATH:-${TRIDENT_HOME}/data/leo_trident.db}"
SERVICE_NAME="${LEO_SERVICE_NAME:-leo-trident.service}"
SKIP_SYSTEMD="${SKIP_SYSTEMD:-0}"


t_start=$(date +%s)
step=0
total=7

pass() {
  step=$((step + 1))
  printf '[%d/%d] %-32s [PASS] %s\n' "${step}" "${total}" "$1" "${2:-}"
}

fail() {
  step=$((step + 1))
  printf '[%d/%d] %-32s [FAIL] %s\n' "${step}" "${total}" "$1" "${2:-}" >&2
  elapsed=$(( $(date +%s) - t_start ))
  printf '\nSMOKE FAILED after %ds (step %d/%d).\n' "${elapsed}" "${step}" "${total}" >&2
  exit 1
}

require() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'smoke.sh: required command not found: %s\n' "$1" >&2
    exit 2
  }
}

require curl
require jq
require python3

# ── [1/7] /health ─────────────────────────────────────────────────────────
health_body=$(curl -s --max-time 15 -w '\n%{http_code}' "${TRIDENT_URL}/health" 2>/dev/null) || \
  fail "health endpoint" "service unreachable at ${TRIDENT_URL}"
health_code=$(printf '%s' "${health_body}" | tail -n1)
health_json=$(printf '%s' "${health_body}" | sed '$d')
[ "${health_code}" = "200" ] || \
  fail "health endpoint" "expected 200, got ${health_code}: ${health_json}"
embedder_kind=$(printf '%s' "${health_json}" | jq -r '.checks.embedder // "unknown"')
case "${embedder_kind}" in
  real) pass "health endpoint" "200 ok, embedder=real" ;;
  stub) fail "health endpoint" "embedder=stub (random vectors — NO-GO for prod)" ;;
  *)    fail "health endpoint" "embedder=${embedder_kind} (expected real)" ;;
esac

# ── [2/7] /stats ──────────────────────────────────────────────────────────
stats_body=$(curl -s --fail --max-time 5 "${TRIDENT_URL}/stats" 2>/dev/null) || \
  fail "stats endpoint" "non-200 from ${TRIDENT_URL}/stats"
chunk_count=$(printf '%s' "${stats_body}" | jq -r '.corpus.asme_chunks // "null"') || \
  fail "stats endpoint" "invalid JSON from /stats"
[ "${chunk_count}" != "null" ] || \
  fail "stats endpoint" "missing corpus.asme_chunks"
pass "stats endpoint" "valid JSON, asme_chunks=${chunk_count}"

# ── [3/7] schema sanity ───────────────────────────────────────────────────
[ -f "${DB_PATH}" ] || fail "schema sanity" "db missing at ${DB_PATH}"
schema_dump=$(python3 - "${DB_PATH}" <<'PY' 2>/dev/null
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
names = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE name IN "
    "('conversation_logs','logs_fts','idx_logs_session','idx_logs_created')"
)]
print("\n".join(names))
PY
) || fail "schema sanity" "sqlite read failed at ${DB_PATH}"
for required in conversation_logs logs_fts idx_logs_session idx_logs_created; do
  printf '%s\n' "${schema_dump}" | grep -qx "${required}" || \
    fail "schema sanity" "missing schema object: ${required}"
done
pass "schema sanity" "conversation_logs + indexes present"

# ── [4/7] log_turn round-trip ─────────────────────────────────────────────
nonce=$(date +%s%N | sha256sum | head -c 16)
keyword="smoketokn${nonce}"
session_id="smoke-${nonce}"
log_payload=$(jq -n --arg s "${session_id}" --arg k "${keyword}" \
  '{session_id:$s, user:("ping " + $k), assistant:("pong " + $k)}')
log_resp=$(curl -s --fail --max-time 5 -H 'Content-Type: application/json' \
  -d "${log_payload}" "${TRIDENT_URL}/log_turn" 2>/dev/null) || \
  fail "log_turn round-trip" "POST /log_turn failed"
[ "$(printf '%s' "${log_resp}" | jq -r '.ok')" = "true" ] || \
  fail "log_turn round-trip" "log_turn ok != true: ${log_resp}"

search_payload=$(jq -n --arg t "${keyword}" --arg s "${session_id}" \
  '{text:$t, top_k:5, session_id:$s}')
search_resp=$(curl -s --fail --max-time 5 -H 'Content-Type: application/json' \
  -d "${search_payload}" "${TRIDENT_URL}/search_conversations" 2>/dev/null) || \
  fail "log_turn round-trip" "POST /search_conversations failed"
hit_count=$(printf '%s' "${search_resp}" | jq -r '.results | length')
[ "${hit_count}" -ge 1 ] 2>/dev/null || \
  fail "log_turn round-trip" "expected >=1 hit for '${keyword}', got ${hit_count}"
pass "log_turn round-trip" "logged + retrieved (${hit_count} hits)"

# ── [5/7] /query ──────────────────────────────────────────────────────────
# Query for the keyword we just logged in step 4 with conversations included.
# This exercises the full /query stack regardless of corpus state.
query_payload=$(jq -n --arg t "${keyword}" \
  '{text:$t, top_k:5, use_rerank:false, include_conversations:true}')
query_resp=$(curl -s --fail --max-time 20 -H 'Content-Type: application/json' \
  -d "${query_payload}" "${TRIDENT_URL}/query" 2>/dev/null) || \
  fail "query endpoint" "POST /query failed"
[ "$(printf '%s' "${query_resp}" | jq -r '.stub_embedder')" = "false" ] || \
  fail "query endpoint" "stub_embedder=true in /query response"
qhits=$(printf '%s' "${query_resp}" | jq -r '.results | length')
[ "${qhits}" -ge 1 ] 2>/dev/null || \
  fail "query endpoint" "expected >=1 hit for '${keyword}', got ${qhits}"
pass "query endpoint" "${qhits} hits for '${keyword}'"

# ── [6/7] ingest_fact round-trip ──────────────────────────────────────────
fact_key="smokefact_${nonce}"
fact_value="canary_${nonce}_unique"
fact_payload=$(jq -n --arg k "${fact_key}" --arg v "${fact_value}" \
  '{category:"smoke", key:$k, value:$v}')
fact_resp=$(curl -s --fail --max-time 10 -H 'Content-Type: application/json' \
  -d "${fact_payload}" "${TRIDENT_URL}/ingest_fact" 2>/dev/null) || \
  fail "ingest_fact round-trip" "POST /ingest_fact failed"
[ "$(printf '%s' "${fact_resp}" | jq -r '.ok')" = "true" ] || \
  fail "ingest_fact round-trip" "ingest_fact ok != true: ${fact_resp}"

# Query for the unique value we just ingested.
fact_query_payload=$(jq -n --arg t "${fact_value}" \
  '{text:$t, top_k:10, use_rerank:false, include_conversations:false}')
fact_query_resp=$(curl -s --fail --max-time 20 -H 'Content-Type: application/json' \
  -d "${fact_query_payload}" "${TRIDENT_URL}/query" 2>/dev/null) || \
  fail "ingest_fact round-trip" "POST /query for fact failed"
fhits=$(printf '%s' "${fact_query_resp}" | jq -r '.results | length')
[ "${fhits}" -ge 1 ] 2>/dev/null || \
  fail "ingest_fact round-trip" "ingested fact not retrievable (0 hits for ${fact_value})"
pass "ingest_fact round-trip" "ingested + retrieved (${fhits} hits)"

# ── [7/7] systemd service ─────────────────────────────────────────────────
if [ "${SKIP_SYSTEMD}" = "1" ]; then
  pass "systemd service" "skipped (SKIP_SYSTEMD=1)"
else
  if ! command -v systemctl >/dev/null 2>&1; then
    fail "systemd service" "systemctl not found (set SKIP_SYSTEMD=1 to skip)"
  fi
  state=$(systemctl --user is-active "${SERVICE_NAME}" 2>/dev/null || true)
  [ "${state}" = "active" ] || \
    fail "systemd service" "${SERVICE_NAME} is '${state}', expected 'active'"
  pass "systemd service" "${SERVICE_NAME} active"
fi

elapsed=$(( $(date +%s) - t_start ))
printf '\nSMOKE PASSED — %d/%d steps in %ds.\n' "${total}" "${total}" "${elapsed}"
exit 0
