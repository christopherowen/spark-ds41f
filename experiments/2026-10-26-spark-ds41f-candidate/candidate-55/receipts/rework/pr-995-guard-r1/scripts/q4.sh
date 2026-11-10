#!/bin/bash
# q4.sh: second confirmation set, ABBA twice (base, cand, cand, base, base, cand, cand, base), boots 3..6
# of the confirmation: per boot, confirm995.py (the gate's copy cells at 201 replays) and, in a second
# fresh container, k32995.py (the two guarded kernels, 32 pairs a graph). One container a step, run.sh.
W=$HOME/spark-lab/agents/round3-995-guard; cd $W
retry() { for i in $(seq 1 30); do [ -e $HOME/spark-hold.json ] && { echo "hold file present: stop"; exit 9; }; ./run.sh "$@"; rc=$?; [ $rc -ne 3 ] && return $rc; sleep 60; done; return 3; }
for spec in "base 3" "cand 3" "cand 4" "base 4" "base 5" "cand 5" "cand 6" "base 6"; do
  set -- $spec
  retry confirm-$1-boot$2 $1 -- python3 /h/confirm995.py
  retry k32-$1-boot$2 $1 -- python3 /h/k32995.py --label $1-boot$2 --out /out/k32-$1-boot$2.jsonl
done
echo "QUEUE4 DONE $(date -u +%FT%TZ)"
