#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# Lean screen, one session: control, drafter head, Markov, both, control again.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-draft-precision
LEAN=(--suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink,explain
      --concurrency 1,8 --min-samples 3 --max-samples 3)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in control:s-control dhead:s-dhead markov:s-markov both:s-both control:s-control-b; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${LEAN[@]}"
  log "arm $arm ($label) exit $?"
done
