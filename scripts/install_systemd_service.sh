#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-bishe-image-classification}"
PROJECT_DIR="${PROJECT_DIR:-/root/.openclaw/workspace-taizi/bishe-image-classification}"
APP_PORT="${APP_PORT:-19001}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-road-admin-2026}"

cat >/etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Graduation Image Classification System
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
Environment=ADMIN_PASSWORD=${ADMIN_PASSWORD}
ExecStart=${PROJECT_DIR}/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ${SERVICE_NAME}
systemctl restart ${SERVICE_NAME}
systemctl --no-pager --full status ${SERVICE_NAME} || true
curl -fsS http://127.0.0.1:${APP_PORT}/api/health
