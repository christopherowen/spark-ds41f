#!/bin/bash
# Diagnosis, second batch (reordered: #1001 screen configs first), on the c51 arm.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; RW=$W/bin/rw.sh; mkdir -p $W/out/pr-diag/json
for spec in "sconf 1" "sconf 2" "screen 1" "screen 2" "vocab 2"; do set -- $spec
  $RW diag d963-$1-boot$2 c51 -- python3 /h/d963.py $1 /out/json/d963-$1-boot$2.json
done
