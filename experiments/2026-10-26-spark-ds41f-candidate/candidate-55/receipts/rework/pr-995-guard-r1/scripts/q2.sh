#!/bin/bash
# q2.sh: the supplementary identity queue (harness/ids995b.py), run after q.sh's QUEUE DONE.
# ABBA: base boot1, cand boot1, cand boot2, base boot2; one fresh container a boot, through run.sh.
W=$HOME/spark-lab/agents/round3-995-guard; cd $W
until grep -q "QUEUE DONE" q.log; do sleep 20; done
retry() { for i in $(seq 1 30); do [ -e $HOME/spark-hold.json ] && { echo "hold file present: stop"; exit 9; }; ./run.sh "$@"; rc=$?; [ $rc -ne 3 ] && return $rc; sleep 60; done; return 3; }
for spec in "base 1" "cand 1" "cand 2" "base 2"; do
  set -- $spec
  retry ids2-$1-boot$2 $1 -- python3 /h/ids995b.py --label $1-boot$2 --out /out/ids2-$1-boot$2.jsonl
done
echo "QUEUE2 DONE $(date -u +%FT%TZ)"
