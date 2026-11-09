#!/bin/bash
# Same-session ABBA on sp4: no1016 (candidate-47 without #1016 = 1107's gemm.py), r1, r1, no1016.
cd $HOME/spark-lab/agents/candidate-47-rework-1016
for step in "no1016 boot1" "r1 boot1" "r1 boot2" "no1016 boot2"; do
  set -- $step
  ./om-r1.sh mark-$1-$2 $1 -- python3 /h/h1016r1.py $1-$2 mark || { echo "FAILED $step rc=$?"; exit 1; }
  grep -E "^(mark|prod|qb|layer)" out/mark-$1-$2.txt | tail -3
done
echo QUEUE-DONE
