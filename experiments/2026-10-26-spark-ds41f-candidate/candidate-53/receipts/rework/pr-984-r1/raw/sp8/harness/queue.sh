#!/bin/bash
# The session: a warm-up boot of the candidate trees (their TileLang kernels), then
# base, fix, c984, c984, fix, base (two boots of each arm, mirrored order).
cd $HOME/spark-lab/agents/rework-984-r1
for spec in warm-fix:fix base-1:base fix-1:fix c984-1:c984 c984-2:c984 fix-2:fix base-2:base; do
  label=${spec%%:*}; arm=${spec##*:}
  harness/run.sh $label $arm
  echo "$(date -u +%FT%TZ) $label $arm rc=$?" >> out/queue.log
done
echo "QUEUE DONE $(date -u +%FT%TZ)" >> out/queue.log
