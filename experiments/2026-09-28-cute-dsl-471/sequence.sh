#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-cute-dsl-471
RUN=(--suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink
     --concurrency 1,8 --min-samples 3 --max-samples 3)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in ${STEPS:-candidate:c-candidate control:c-control candidate-nol2:c-nol2 candidate:c-candidate-b control:c-control-b candidate-nol2:c-nol2-b}; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${RUN[@]}"
  log "arm $arm ($label) exit $?"
done
