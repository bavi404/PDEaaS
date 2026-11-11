#!/usr/bin/env bash
# Emulate Render Free (512 MB RAM, <1 CPU) with the unchanged M6 image. No swap.
FIELD=$(python3 -c 'import json;print(json.dumps({"permeability":[[1.0 if (i//16+j//16)%2 else 0.1 for j in range(128)] for i in range(128)]}))')
for cpus in 0.5 0.1; do
  docker rm -f render-sim >/dev/null 2>&1
  t0=$(date +%s%N)
  docker run -d --name render-sim --cpus=$cpus --memory=512m --memory-swap=512m -p 7860:7860 pdeflow:m6 >/dev/null
  ok=0
  for i in $(seq 600); do
    if curl -sf -o /dev/null --max-time 5 http://127.0.0.1:7860/api/health; then ok=1; break; fi
    [ "$(docker inspect -f '{{.State.Running}}' render-sim)" = false ] && break
    sleep 0.5
  done
  t1=$(date +%s%N)
  echo "== --cpus=$cpus --memory=512m (no swap): healthy=$ok after $(( (t1-t0)/1000000 )) ms; state: $(docker inspect -f 'running={{.State.Running}} oom_killed={{.State.OOMKilled}} exit={{.State.ExitCode}}' render-sim)"
  if [ $ok = 1 ]; then
    for k in 1 2 3; do curl -s -o /dev/null -w "   sample 4701 128 default: HTTP %{http_code} %{time_total}s\n" --max-time 120 -H 'content-type: application/json' -d '{"sample_id":"test-4701","resolution":128}' http://127.0.0.1:7860/api/predict; done
    for k in 1 2; do curl -s -o /dev/null -w "   custom 128 field: HTTP %{http_code} %{time_total}s\n" --max-time 120 -H 'content-type: application/json' -d "$FIELD" http://127.0.0.1:7860/api/predict; done
    curl -s -o /dev/null -w "   GET / (frontend): HTTP %{http_code} %{time_total}s\n" --max-time 60 http://127.0.0.1:7860/
    docker exec render-sim sh -c 'echo "   cgroup peak $(( $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes) / 1048576 )) MiB of 512; limit hits (failcnt) $(cat /sys/fs/cgroup/memory/memory.failcnt); state after: '"'"'$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)'"'"' bytes"'
    echo "   container still running: $(docker inspect -f '{{.State.Running}} oom_killed={{.State.OOMKilled}}' render-sim)"
  fi
  docker logs render-sim 2>&1 | grep -E '"level": ?"ERROR"|Killed|MemoryError' | head -3
done
docker rm -f render-sim >/dev/null
