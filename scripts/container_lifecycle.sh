#!/usr/bin/env bash
# Container lifecycle checks (docs/DOCKER.md §4): startup time, graceful stop, checkpoint tampering, process death.
#   scripts/container_lifecycle.sh [image]        (default image: pdeflow:m6)
set -uo pipefail
IMG="${1:-pdeflow:m6}"
TMP=$(mktemp -d)
fail=0
ok() { echo "PASS  $*"; }
ko() { echo "FAIL  $*"; fail=1; }
health() { docker exec "$1" python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=2)" 2>/dev/null; }

echo "== startup time and graceful stop (2 vCPU)"
docker rm -f lc-stop >/dev/null 2>&1
t0=$(date +%s%N); docker run -d --name lc-stop --cpus=2 "$IMG" >/dev/null
for _ in $(seq 240); do health lc-stop && break; sleep 0.25; done
t1=$(date +%s%N); echo "      ready (public /api/health 200) after $(( (t1 - t0) / 1000000 )) ms"
t0=$(date +%s%N); docker stop -t 30 lc-stop >/dev/null; t1=$(date +%s%N)
code=$(docker inspect -f '{{.State.ExitCode}}' lc-stop)
logs=$(docker logs lc-stop 2>&1)
if [[ $code == 0 ]] && grep -q '"event": "shutdown"' <<<"$logs" && grep -q '"event":"stopped"' <<<"$logs"; then
  ok "docker stop: exit 0 in $(( (t1 - t0) / 1000000 )) ms; API lifespan shutdown logged"
else ko "docker stop: exit $code"; fi
docker rm lc-stop >/dev/null

echo "== tampered checkpoint is refused before anything starts"
head -c 4096 /dev/urandom > "$TMP/bad.pt"
docker run --name lc-bad -v "$TMP/bad.pt:/app/backend/models/fno_darcy_beta1_baseline.pt:ro" "$IMG" >/dev/null 2>"$TMP/bad.log"
code=$(docker inspect -f '{{.State.ExitCode}}' lc-bad); docker rm lc-bad >/dev/null
if [[ $code == 3 ]] && grep -q '"level":"ERROR".*checkpoint_sha256_mismatch' "$TMP/bad.log"; then ok "exit 3, ERROR checkpoint_sha256_mismatch"; else ko "tampered checkpoint: exit $code"; cat "$TMP/bad.log"; fi

for proc in uvicorn next-server; do
  echo "== $proc dies -> container exits non-zero (platform restarts it)"
  docker rm -f lc-crash >/dev/null 2>&1
  docker run -d --name lc-crash "$IMG" >/dev/null
  for _ in $(seq 240); do health lc-crash && break; sleep 0.25; done
  docker exec lc-crash python -c "
import os, signal
for p in os.listdir('/proc'):
    if p.isdigit() and int(p) != os.getpid():
        try: cmd = open(f'/proc/{p}/cmdline','rb').read().replace(b'\0', b' ').decode()
        except OSError: continue
        if '$proc' in cmd and 'python -c' not in cmd:
            os.kill(int(p), signal.SIGKILL); print('killed', p, cmd[:60])"
  for _ in $(seq 40); do [[ $(docker inspect -f '{{.State.Running}}' lc-crash) == false ]] && break; sleep 0.25; done
  running=$(docker inspect -f '{{.State.Running}}' lc-crash); code=$(docker inspect -f '{{.State.ExitCode}}' lc-crash)
  if [[ $running == false && $code != 0 ]]; then ok "$proc killed: container stopped, exit $code"; else ko "$proc killed: running=$running exit=$code"; fi
  docker logs lc-crash 2>&1 | grep process_exited
  docker rm -f lc-crash >/dev/null
done
rm -rf "$TMP"
exit $fail
