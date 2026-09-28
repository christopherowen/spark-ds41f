#!/bin/bash
# usage: build.sh   (on dgx1, deployment checkout at this experiment's commit; cluster stopped)
# Builds the r5i image and copies it to the other nodes.
set -eu
cd ~/projects/spark3-vllm-ds41f
TAG=vllm-ds41f-kkref:04c30fa98e79-r5i
log() { echo "$(date -u +%FT%TZ) $*"; }
for _ in $(seq 1 60); do
  avail=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
log "build prepare (MemAvailable ${avail} GiB)"
bin/spark3 build prepare
log "build image $TAG"
bin/spark3 build image --apply --tag "$TAG"
local_id=$(docker image inspect "$TAG" --format "{{.Id}}")
for host in dgx2 dgx3; do
  log "copy image to $host"
  docker save "$TAG" | ssh "$host" docker load
  remote_id=$(ssh "$host" docker image inspect "$TAG" --format "{{.Id}}")
  [ "$remote_id" = "$local_id" ] || { log "image ID differs on $host: $remote_id"; exit 1; }
done
log "image $local_id on all nodes"
