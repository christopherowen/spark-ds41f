#!/bin/bash
# q3.sh: confirmation of the gate's copy cells at 201 replays (harness/confirm995.py), ABBA:
# base boot1, cand boot1, cand boot2, base boot2; one fresh container a boot, through run.sh.
W=$HOME/spark-lab/agents/round3-995-guard; cd $W
retry() { for i in $(seq 1 30); do [ -e $HOME/spark-hold.json ] && { echo "hold file present: stop"; exit 9; }; ./run.sh "$@"; rc=$?; [ $rc -ne 3 ] && return $rc; sleep 60; done; return 3; }
for spec in "base 1" "cand 1" "cand 2" "base 2"; do
  set -- $spec
  retry confirm-$1-boot$2 $1 -- python3 /h/confirm995.py
done
echo "QUEUE3 DONE $(date -u +%FT%TZ)"
