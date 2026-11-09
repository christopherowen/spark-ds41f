#!/bin/bash
# Diagnosis (in-process variants, d963.py) on the c51 arm, two boots per section.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; RW=$W/bin/rw.sh; mkdir -p $W/out/diag/json
for sec in "$@"; do
  for k in 1 2; do
    $RW diag d963-$sec-boot$k c51 -- python3 /h/d963.py $sec /out/json/d963-$sec-boot$k.json
  done
done
