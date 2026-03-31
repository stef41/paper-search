#!/bin/bash
set -e

HOSTNAME="${PILOT_HOSTNAME:-arxiv-search}"
EMAIL="${PILOT_EMAIL:-arxiv-search@pilotprotocol.network}"

echo "[pilot] initialising config …"
pilotctl init --hostname "$HOSTNAME" 2>/dev/null || true

echo "[pilot] starting daemon (foreground=false) …"
pilotctl daemon start \
    --hostname "$HOSTNAME" \
    --email "$EMAIL" \
    --public

echo "[pilot] waiting for daemon …"
for i in $(seq 1 30); do
    if pilotctl daemon status --check 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "[pilot] setting tags …"
pilotctl set-tags arxiv api paper-search || true

echo "[pilot] ensuring public visibility …"
pilotctl set-public || true

INFO=$(pilotctl --json info 2>/dev/null || true)
echo "[pilot] daemon info: $INFO"

echo "[pilot] starting HTTP proxy → ${UPSTREAM_HOST:-api}:${UPSTREAM_PORT:-8000}"
exec python /app/pilot_proxy.py
