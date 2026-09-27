#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# control, tile32, simple, control again: real-text prefill plus single-stream decode.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-prefill-moe-tile
RUN=(--suites quality,decode,prefill --decode-cases prose-nothink,code-nothink,explain
     --concurrency 1 --min-samples 3 --max-samples 3
     --prefill-text source --prefill-sizes 4096,16384,32768,65536 --prefill-repeats 2)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in control:s-control tile32:s-tile32 simple:s-simple control:s-control-b; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${RUN[@]}"
  log "arm $arm ($label) exit $?"
done
