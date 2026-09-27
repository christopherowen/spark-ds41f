#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-prefill-sp
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in control:s-control sp:s-sp control:s-control-b; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label"
  log "arm $arm ($label) exit $?"
done
