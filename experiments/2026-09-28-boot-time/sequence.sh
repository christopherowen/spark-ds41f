#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-boot-time
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in ${STEPS:-base:b-base-1 base:b-base-2 persist:b-persist-1 persist:b-persist-2 persist:b-persist-3:pyspy nccl-info:b-nccl-info nogin:b-nogin}; do
  IFS=: read -r arm label mode <<< "$step"
  log "boot $arm ($label) $mode"
  $E/boot.sh "$arm" "$label" $mode
  log "boot $arm ($label) exit $?"
done
python3 $E/timeline.py results/private/boot/b-*
