#!/usr/bin/env bash
# The corrected trees' own tests (PR #979's files), GPU and CPU suites, back to back.
W=$HOME/spark-lab/agents/candidate-49-rework-979-r1
for arm in "$@"; do
  for kind in gpu cpu; do "$W/runtest.sh" "test-$kind-$arm-$(hostname)" "$arm" "$kind"; done
done
