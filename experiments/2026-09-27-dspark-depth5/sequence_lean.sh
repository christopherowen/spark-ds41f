#!/bin/bash
# usage: sequence_lean.sh   (on dgx1, deployment checkout at this experiment's commit)
# Screening pass, as in experiments/2026-09-27-lil-head/sequence_lean.sh.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-dspark-depth5
LEAN=(--suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink
      --concurrency 1,8 --min-samples 3 --max-samples 3)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in ${STEPS:-k3real:s-k3real k5real:s-k5real}; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${LEAN[@]}"
  log "arm $arm ($label) exit $?"
done
