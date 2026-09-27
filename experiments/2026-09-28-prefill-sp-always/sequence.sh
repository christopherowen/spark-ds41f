#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# current, always, current, always: small real-text prefills plus decode.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-prefill-sp-always
RUN=(--suites quality,decode,prefill --decode-cases prose-nothink,code-nothink
     --concurrency 1,8 --min-samples 3 --max-samples 3
     --prefill-text source --prefill-sizes 64,128,256,512,1024,1536,2048,4096 --prefill-repeats 4)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in ${STEPS:-current:s-current always:s-always current:s-current-b always:s-always-b}; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${RUN[@]}"
  log "arm $arm ($label) exit $?"
done
