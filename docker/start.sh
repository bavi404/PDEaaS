#!/usr/bin/env bash
# PDEFlow container entrypoint: one Docker Space running two processes.
#   FastAPI/uvicorn  127.0.0.1:8000  (internal only; serves the pinned FNO checkpoint)
#   Next.js server   0.0.0.0:${PORT} (public; proxies /api/* to FastAPI)
# 1. Verify the checkpoint SHA-256 before anything starts (the API verifies it again when it loads).
# 2. Start the API and wait until /health answers (startup fails fast if it cannot load).
# 3. Start the web server.
# 4. SIGTERM/SIGINT are forwarded to both processes (graceful shutdown); if either exits, the other is stopped
#    and the container exits with that status, so the platform restarts it.
set -Eeuo pipefail

# log EVENT [JSON_FIELDS] [LEVEL]
log() { printf '{"ts":"%s","level":"%s","logger":"pdeflow.entrypoint","event":"%s"%s}\n' "$(date -u +%FT%T.%3N+00:00)" "${3:-INFO}" "$1" "${2:-}" >&2; }

read -r EXPECTED CHECKPOINT < <(python -c 'from pdeflow.api.settings import Settings; s = Settings(); print(s.expected_sha256, s.checkpoint)')
ACTUAL=$(sha256sum "$CHECKPOINT" | cut -d" " -f1)
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  log "checkpoint_sha256_mismatch" ",\"expected\":\"$EXPECTED\",\"actual\":\"$ACTUAL\"" ERROR
  exit 3
fi
log "checkpoint_verified" ",\"sha256\":\"$ACTUAL\",\"path\":\"$CHECKPOINT\""

API_PORT="${PDEFLOW_API_PORT:-8000}"
uvicorn pdeflow.api.app:app --host 127.0.0.1 --port "$API_PORT" --no-access-log --timeout-graceful-shutdown 10 &
API_PID=$!

shutdown() {
  log "shutdown_requested" ",\"signal\":\"$1\""
  kill -TERM "${WEB_PID:-}" "$API_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  log "stopped"
  exit 0
}
trap 'shutdown TERM' TERM
trap 'shutdown INT' INT

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" || code=$?
    log "api_failed_to_start" ",\"exit_code\":${code:-1}" ERROR
    exit "${code:-1}"
  fi
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$API_PORT/health', timeout=2)" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "$ready" != 1 ]]; then
  log "api_not_ready_after_60s" "" ERROR
  kill -TERM "$API_PID" 2>/dev/null || true
  exit 1
fi
log "api_ready" ",\"port\":$API_PORT"

PDEFLOW_API_URL="http://127.0.0.1:$API_PORT" node /app/web/server.js &
WEB_PID=$!
log "web_started" ",\"port\":${PORT:-7860}"

# exit as soon as either process stops; stop the other one too
set +e
wait -n "$API_PID" "$WEB_PID"
code=$?
log "process_exited" ",\"exit_code\":$code" "$([[ $code == 0 ]] && echo INFO || echo ERROR)"
kill -TERM "$API_PID" "$WEB_PID" 2>/dev/null
wait
exit "$code"
