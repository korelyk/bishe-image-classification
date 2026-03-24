#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-19001}"
HOST="${HOST:-0.0.0.0}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-road-admin-2026}"

cd "$ROOT_DIR"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install -r requirements.txt >/dev/null

echo "[1/2] starting uvicorn on ${HOST}:${PORT} ..."
pkill -f "uvicorn app.main:app --host ${HOST} --port ${PORT}" || true
nohup env ADMIN_PASSWORD="$ADMIN_PASSWORD" .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" > /tmp/bishe-uvicorn.log 2>&1 &
sleep 3
curl -fsS "http://${HOST}:${PORT}/api/health" >/dev/null

echo "[2/2] starting public tunnel ..."
pkill -f "localtunnel --port ${PORT}" || true
nohup npx --yes localtunnel --port "$PORT" > /tmp/bishe-localtunnel.log 2>&1 &
sleep 8
URL="$(grep -Eo 'https://[^ ]+\.loca\.lt' /tmp/bishe-localtunnel.log | tail -1 || true)"
if [ -z "$URL" ]; then
  echo "failed to get public url"
  exit 1
fi

echo "public url: $URL"
echo "admin password: $ADMIN_PASSWORD"
