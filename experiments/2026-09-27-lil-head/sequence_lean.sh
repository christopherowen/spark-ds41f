#!/bin/bash
# usage: sequence_lean.sh   (on dgx1, deployment checkout at this experiment's commit)
# Screening pass: one boot per arm, reasoning and answer cases at one and
# eight streams, three samples each. Single-stream step time and accepted
# drafts per step decide; a candidate gets the full matrix before promotion.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-lil-head
P=experiments/2026-09-26-dspark-policy
LEAN=(--suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink
      --concurrency 1,8 --min-samples 3 --max-samples 3)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in ${STEPS:-base:s-base realprof:s-realprof k5real:s-k5real k5:s-k5 nol2:s-nol2}; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${LEAN[@]}"
  log "arm $arm ($label) exit $?"
done
if [ -z "${STEPS:-}" ]; then
  log "arm det (s-det)"
  $P/run_arm.sh det s-det "${LEAN[@]}"
  log "arm det (s-det) exit $?"
fi
