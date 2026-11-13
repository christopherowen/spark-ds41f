#!/bin/bash
# #977 confirmation queue (ABBA): the PR bench at <=32K with 201 replays a cell.
B=$HOME/spark-lab/agents/candidate-53-own-marks; cd $B
M="-v $B/src/pr-977/experiments/2026-11-08-copy-drafter-long-context-scan:/exp:ro -v $B/harness:/h:ro"
for spec in "base base 1" "c53 candidate 1" "c53 candidate 2" "base base 2"; do
  set -- $spec
  for i in $(seq 1 30); do [ -e $HOME/spark-hold.json ] && exit 9; common/run.sh 977 confirm-$1-boot$3 $1 $M -- python3 /h/warm.py /h/confirm977.py /exp/bench_copy_match_long_context.py --label $2 --out /out/confirm-$1-boot$3.jsonl; rc=$?; [ $rc -ne 3 ] && break; sleep 60; done
done
echo "CONFIRM DONE $(date -u +%FT%TZ)"
