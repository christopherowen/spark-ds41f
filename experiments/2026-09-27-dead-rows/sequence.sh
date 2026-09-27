#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# Lean screen, one session: control, tau 0.3, tau 0.5, control again.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-dead-rows
LEAN=(--suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink
      --concurrency 1,8 --min-samples 3 --max-samples 3)
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in control:s-control dead03:s-dead03 dead05:s-dead05 control:s-control-b; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" "${LEAN[@]}"
  log "arm $arm ($label) exit $?"
done
