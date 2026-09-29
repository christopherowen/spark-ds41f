#!/bin/bash
# usage: build.sh   (on dgx1, deployment checkout at this experiment's commit)
# Stops every service, builds the r5l candidate image from the lock (vLLM
# 0001-0026, B12X f8069b2c + 0001-0002), and loads the same image on dgx2/3.
set -euo pipefail
cd ~/projects/spark3-vllm-ds41f
TAG=vllm-ds41f-kkref:04c30fa98e79-r5l
log() { echo "$(date -u +%FT%TZ) $*"; }
# Refuse before stopping anything: bin/spark3 build never overwrites a tag.
if docker image inspect "$TAG" >/dev/null 2>&1; then
  log "$TAG already exists; remove it or pick a new tag"
  exit 1
fi
for config in config/cluster.json experiments/2026-09-29-r5l/cluster-*.json \
    experiments/2026-09-29-indexer-split/cluster-*.json experiments/2026-09-29-r5k/cluster-*.json; do
  bin/spark3 --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark3 build prepare
bin/spark3 build image --apply --tag "$TAG"
local_id=$(docker image inspect "$TAG" --format "{{.Id}}")
for host in dgx2 dgx3; do
  docker save "$TAG" | ssh "$host" docker load
  [ "$(ssh "$host" docker image inspect "$TAG" --format "{{.Id}}")" = "$local_id" ] || { log "image differs on $host"; exit 1; }
done
log "image $local_id on all nodes"
