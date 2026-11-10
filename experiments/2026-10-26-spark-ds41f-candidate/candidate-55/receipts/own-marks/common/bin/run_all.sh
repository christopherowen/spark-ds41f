#!/bin/bash
# Every run, in this order; each run takes the GPU lock on its own (bin/om.sh), so other
# agents' jobs interleave between runs. progress.log records each script's start and end.
B=$HOME/spark-lab/agents/round3-own-marks
for s in "$@"; do
  echo "$(date -u +%FT%TZ) start $s" >> $B/progress.log
  bash $B/bin/$s.sh; echo "$(date -u +%FT%TZ) end $s rc=$?" >> $B/progress.log
done
