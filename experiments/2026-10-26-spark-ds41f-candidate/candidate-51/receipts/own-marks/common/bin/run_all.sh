#!/bin/bash
# The PR batches back to back; progress in out/progress.log.
B=$HOME/spark-lab/agents/candidate-51-own-marks
for pr in "$@"; do
  echo "$(date -u +%FT%TZ) start pr-$pr" >> $B/out/progress.log
  bash $B/bin/pr-$pr.sh >> $B/out/progress.log 2>&1
  echo "$(date -u +%FT%TZ) end pr-$pr" >> $B/out/progress.log
done
echo "$(date -u +%FT%TZ) all done" >> $B/out/progress.log
