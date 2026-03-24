#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-45.136.12.228}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/opt/bishe-image-classification}"
SERVICE_NAME="${SERVICE_NAME:-bishe-image-classification}"
APP_PORT="${APP_PORT:-19001}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-road-admin-2026}"
REMOTE_PASSWORD="${REMOTE_PASSWORD:-}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_TAR="/tmp/${SERVICE_NAME}.tar.gz"

rm -f "$TMP_TAR"
tar --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='data/app.db' --exclude='data/uploads' --exclude='data/annotated' -czf "$TMP_TAR" -C "$LOCAL_DIR" .

SSH_BASE=(ssh -o StrictHostKeyChecking=no -p "$REMOTE_PORT")
SCP_BASE=(scp -o StrictHostKeyChecking=no -P "$REMOTE_PORT")
if [ -n "$REMOTE_PASSWORD" ]; then
  SSH_BASE=(sshpass -e ssh -o StrictHostKeyChecking=no -p "$REMOTE_PORT")
  SCP_BASE=(sshpass -e scp -o StrictHostKeyChecking=no -P "$REMOTE_PORT")
  export SSHPASS="$REMOTE_PASSWORD"
fi

"${SCP_BASE[@]}" "$TMP_TAR" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/${SERVICE_NAME}.tar.gz"

"${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "bash -s" <<EOF
set -euo pipefail
mkdir -p '${REMOTE_DIR}'
find '${REMOTE_DIR}' -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf {} +
tar -xzf '/tmp/${SERVICE_NAME}.tar.gz' -C '${REMOTE_DIR}'
cd '${REMOTE_DIR}'
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
mkdir -p data/uploads data/annotated data/models
cat >/etc/systemd/system/${SERVICE_NAME}.service <<SYSTEMD
[Unit]
Description=Graduation Image Classification System
After=network.target

[Service]
Type=simple
WorkingDirectory=${REMOTE_DIR}
Environment=ADMIN_PASSWORD=${ADMIN_PASSWORD}
ExecStart=${REMOTE_DIR}/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SYSTEMD
systemctl daemon-reload
systemctl enable --now ${SERVICE_NAME}.service
systemctl restart ${SERVICE_NAME}.service
sleep 3
systemctl --no-pager --full status ${SERVICE_NAME}.service || true
curl -fsS http://127.0.0.1:${APP_PORT}/api/health
EOF

echo "deploy finished: http://${REMOTE_HOST}:${APP_PORT}"
