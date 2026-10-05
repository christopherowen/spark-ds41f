#!/bin/bash
# usage: build.sh   (on dgx1, deployment checkout at this experiment's commit)
# Stops every service (the build needs most of dgx1's memory), builds the r5n
# candidate image from the lock (vLLM 0001-0026, B12X f8069b2c + 0001-0004) and
# loads the same image on dgx2/3.
set -euo pipefail
cd ~/projects/spark3-vllm-ds41f
TAG=vllm-ds41f-kkref:04c30fa98e79-r5n
log() { echo "$(date -u +%FT%TZ) $*"; }
if docker image inspect "$TAG" >/dev/null 2>&1; then
  log "$TAG already exists; remove it or pick a new tag"
  exit 1
fi
for config in config/cluster.json experiments/2026-09-30-r5n/cluster-*.json; do
  bin/spark --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark build prepare
bin/spark build image --apply --tag "$TAG"
local_id=$(docker image inspect "$TAG" --format "{{.Id}}")
for host in dgx2 dgx3; do
  docker save "$TAG" | ssh "$host" docker load
  [ "$(ssh "$host" docker image inspect "$TAG" --format "{{.Id}}")" = "$local_id" ] || { log "image differs on $host"; exit 1; }
done
log "image $local_id on all nodes"
