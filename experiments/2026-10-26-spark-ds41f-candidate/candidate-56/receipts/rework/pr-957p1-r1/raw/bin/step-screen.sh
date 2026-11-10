#!/bin/bash
# 0795-r1 at TP4 rank shapes on one GPU (step957.py, round 3's own-mark harness), one session:
# base, r1 (COMP default), r1c0 (COMP_MB=0), r1c0, r1, base. Two boots an arm.
W=$HOME/spark-lab/agents/candidate-56-rework-957p1; OM=$W/bin/om.sh; . $W/bin/env.sh
for spec in "base base 1" "r1 r1 1" "r1c0 r1 1" "r1c0 r1 2" "r1 r1 2" "base base 2"; do set -- $spec
  extra=(); [ $1 = r1c0 ] && extra=(-e VLLM_DS41_L2_PREFETCH_COMP_MB=0)
  MINMEM_GIB=95 $OM step-$1-boot$3 $2 "${RECIPE_ENV[@]}" "${extra[@]}" "${HUB[@]}" -- \
    python3 /h/step957.py --out /out/json/step-$1-boot$3.json
done
