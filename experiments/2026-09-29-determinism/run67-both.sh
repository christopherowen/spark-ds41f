#!/bin/bash
# usage: run67-both.sh   (on dgx1, deployment checkout at this experiment's commit)
# Both mHC options through run67, each with its own r5o arm: the TF32 projection with 40 K slices
# (ref4e-s40-b4144, results rec5a) and then, only if the restored cluster is idle, the native route with
# 16 rows per CTA (ref4d-b4144, results rec5b).
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
REC=rec5a bash $E/run67.sh ref4e-s40-b4144 4096
log "run67 ref4e-s40-b4144 exit $?"
sleep 30
m=$(curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' | awk '{s += $2} END {print s + 0}')
[ "$m" = "0" ] || { log "cluster not idle ($m requests); not starting the second validation"; exit 1; }
REC=rec5b bash $E/run67.sh ref4d-b4144 4096
log "run67 ref4d-b4144 exit $?"
