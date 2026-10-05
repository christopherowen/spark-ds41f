#!/bin/bash
# usage: candidate.sh   (on dgx1, deployment checkout at this experiment's commit)
# Builds the r5g image (patches 0001-0017) and copies it to the other nodes,
# then starts cluster-candidate.json and runs the reference bench into
# results/private/bench/r5g-reference. The candidate stays up.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-prefill-sp
TAG=vllm-ds41f-kkref:04c30fa98e79-r5g
log() { echo "$(date -u +%FT%TZ) $*"; }
for config in config/cluster.json $E/cluster-*.json experiments/2026-09-27-prefill-moe-tile/cluster-*.json; do
  bin/spark --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
for _ in $(seq 1 60); do
  avail=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
log "build prepare (MemAvailable ${avail} GiB)"
bin/spark build prepare
log "build image $TAG"
bin/spark build image --apply --tag "$TAG"
local_id=$(docker image inspect "$TAG" --format "{{.Id}}")
for host in dgx2 dgx3; do
  log "copy image to $host"
  docker save "$TAG" | ssh "$host" docker load
  remote_id=$(ssh "$host" docker image inspect "$TAG" --format "{{.Id}}")
  [ "$remote_id" = "$local_id" ] || { log "image ID differs on $host: $remote_id"; exit 1; }
done
log "image $local_id on all nodes"
bin/spark --cluster-config $E/cluster-candidate.json cluster start --replace --apply | grep -v "docker run"
bin/spark --cluster-config $E/cluster-candidate.json doctor --live || true
bin/spark --cluster-config $E/cluster-candidate.json bench --allow-mismatch --compare none \
  --suites quality,decode,prefill,prefix,admission \
  --decode-cases prose,code,prose-nothink,code-nothink,json-nothink \
  --output results/private/bench/r5g-reference
log "bench exit $?"
bin/spark --cluster-config $E/cluster-candidate.json bench --allow-mismatch --compare none --suites prefill \
  --prefill-text source --prefill-sizes 4096,16384,32768,65536 --prefill-repeats 2 \
  --output results/private/bench/r5g-prefill-source
log "real-text prefill exit $?"
