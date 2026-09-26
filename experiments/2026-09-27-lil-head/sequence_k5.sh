#!/bin/bash
# usage: sequence_k5.sh   (on dgx1, deployment checkout at this experiment's commit)
# Five drafts on the rebased stack: the startup profile as is, then the
# distinct-token profile at cost scale 1, twice each, alternating.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-lil-head
CASES=prose,code,prose-nothink,code-nothink,json-nothink
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in k5:k1 k5real:q1 k5:k2 k5real:q2; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" --suites quality,decode --decode-cases "$CASES"
  log "arm $arm ($label) exit $?"
done
