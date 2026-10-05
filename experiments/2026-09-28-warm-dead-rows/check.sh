#!/bin/bash
# usage: check.sh   (on dgx1, deployment checkout at this experiment's commit)
# Builds r5j, distributes it, starts the candidate and checks the gates. The
# candidate stays up.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-warm-dead-rows
TAG=vllm-ds41f-kkref:04c30fa98e79-r5j
log() { echo "$(date -u +%FT%TZ) $*"; }
for config in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark build prepare || exit 1
bin/spark build image --apply --tag "$TAG" || exit 1
local_id=$(docker image inspect "$TAG" --format "{{.Id}}")
for host in dgx2 dgx3; do
  docker save "$TAG" | ssh "$host" docker load
  [ "$(ssh "$host" docker image inspect "$TAG" --format "{{.Id}}")" = "$local_id" ] || { log "image differs on $host"; exit 1; }
done
log "image $local_id on all nodes"
bin/spark --cluster-config $E/cluster-candidate.json cluster start --replace --apply | grep -v "docker run" || exit 1
python3 $E/first_request.py http://10.0.1.71:8000
jit=0
for n in dgx1 dgx2 dgx3; do
  c=$(ssh -n $n "docker logs dsv41-karmic-kraken 2>&1 | grep -cE 'JIT compilation during inference|TileLang begins to compile'")
  echo "$n serving-time compilations: $c"
  jit=$((jit + c))
done
bin/spark --cluster-config $E/cluster-candidate.json bench --allow-mismatch --compare none --suites quality \
  --output results/private/bench/r5j-quality
log "gates: serving-time compilations $jit"
