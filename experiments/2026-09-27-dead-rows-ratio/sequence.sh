#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# Lean screen: ratio twice. The controls are the dead-rows experiment's
# three r5d controls, from the same day and protocol.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-dead-rows-ratio
LEAN=(--suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink
      --concurrency 1,8 --min-samples 3 --max-samples 3)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in ratio:s-ratio ratio:s-ratio-b; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${LEAN[@]}"
  log "arm $arm ($label) exit $?"
done
