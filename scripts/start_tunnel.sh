#!/usr/bin/env bash
# Start a serveo.net SSH tunnel to expose the API publicly.
# Uses a persistent custom subdomain tied to the machine's SSH key.
set -euo pipefail

SUBDOMAIN="${SERVEO_SUBDOMAIN:-arxiv-paperpilot}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
TUNNEL_URL="https://${SUBDOMAIN}.serveousercontent.com"
TUNNEL_LOG="/tmp/serveo.log"

# Kill any existing serveo tunnel
pkill -f "ssh.*serveo.net" 2>/dev/null || true
sleep 1

# Start tunnel in background with keepalive
nohup ssh -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -T -R "${SUBDOMAIN}:80:localhost:${LOCAL_PORT}" serveo.net \
  > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!
echo "Tunnel PID: $TUNNEL_PID"

# Wait for confirmation (up to 15s)
for i in $(seq 1 15); do
    if grep -q "Forwarding HTTP traffic" "$TUNNEL_LOG" 2>/dev/null; then
        echo "Tunnel URL: $TUNNEL_URL"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Tunnel not established after 15s. Check $TUNNEL_LOG" >&2
cat "$TUNNEL_LOG" >&2
exit 1
