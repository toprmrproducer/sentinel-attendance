#!/bin/bash
# Sentinel Attendance deploy script. Run on a fresh Ubuntu 24.04 droplet as root.
# Usage: SERVER_IP=1.2.3.4 ALERT_NUMBER=919307512816 bash setup.sh
set -e

SERVER_IP="${SERVER_IP:?set SERVER_IP}"
ALERT_NUMBER="${ALERT_NUMBER:-}"
PUBLIC_HOST="${SERVER_IP//./-}.sslip.io"

apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git ffmpeg libgl1 libglib2.0-0 curl

mkdir -p /opt/sentinel-attendance
cd /opt/sentinel-attendance
if [ -d .git ]; then
  git pull
else
  git clone https://github.com/toprmrproducer/sentinel-attendance.git .
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -q --upgrade pip
pip install --no-cache-dir -q -r requirements.txt

mkdir -p sample_footage/cafe sample_footage/long_feed data

# systemd service
cat > /etc/systemd/system/sentinel.service << EOF
[Unit]
Description=Sentinel Attendance
After=network.target

[Service]
WorkingDirectory=/opt/sentinel-attendance
Environment="PUBLIC_BASE_URL=https://${PUBLIC_HOST}"
Environment="SENTINEL_ALERT_CALL_NUMBER=${ALERT_NUMBER}"
ExecStart=/opt/sentinel-attendance/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8811
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sentinel
systemctl restart sentinel

# Caddy for automatic HTTPS via sslip.io, same pattern as the Dograh boxes
if ! command -v caddy >/dev/null; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy
fi

cat > /etc/caddy/Caddyfile << EOF
${PUBLIC_HOST} {
    reverse_proxy 127.0.0.1:8811
}
EOF
systemctl reload caddy || systemctl restart caddy

echo "Deployed. Public URL: https://${PUBLIC_HOST}"
