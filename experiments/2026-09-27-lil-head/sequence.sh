#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# Alternates the rebased stack with and without L2 weight prefetch.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-lil-head
CASES=prose,code,prose-nothink,code-nothink,json-nothink
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in base:b1 nol2:n1 base:b2 nol2:n2; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" --suites quality,decode --decode-cases "$CASES"
  log "arm $arm ($label) exit $?"
done
