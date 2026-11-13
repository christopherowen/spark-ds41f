#!/bin/bash
# run_all.sh: each queue.sh section inside one hold of the shared GPU lock, sections in order; the lock is
# released between sections so other agents' jobs can run. Refuses when ~/spark-hold.json exists.
B=$HOME/spark-lab/agents/candidate-52-own-marks
for sec in ${@:-common 1030 968 969}; do
  [ -e $HOME/spark-hold.json ] && { echo "hold file present, stop before $sec"; exit 3; }
  echo "section $sec issued $(date -u +%FT%TZ)"
  OM_LOCK_HELD=1 flock /tmp/spark-lab-gpu.lock bash $B/queue.sh $sec; rc=$?
  echo "section $sec ended $(date -u +%FT%TZ) rc=$rc"
done
echo "queue all done $(date -u +%FT%TZ)"
