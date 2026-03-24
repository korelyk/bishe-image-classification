#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${1:-http://127.0.0.1:19001}"

echo "[1/3] health check: ${BASE_URL}/api/health"
curl -fsS "${BASE_URL}/api/health" | python3 -m json.tool

echo "[2/3] history check"
curl -fsS "${BASE_URL}/api/history?limit=3" | python3 -m json.tool >/dev/null

echo "[3/3] admin stats check"
if [ -n "${ADMIN_PASSWORD:-}" ]; then
  curl -fsS -H "X-Admin-Password: ${ADMIN_PASSWORD}" "${BASE_URL}/api/admin/stats" | python3 -m json.tool >/dev/null
else
  curl -fsS "${BASE_URL}/api/admin/stats" | python3 -m json.tool >/dev/null || true
fi

echo "smoke test passed"
