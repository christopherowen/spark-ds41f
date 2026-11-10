#!/bin/bash
# The mark: 0795-r1 (final, COMP default) at TP4 rank shapes on one GPU (step957.py), ABBA in one
# session against candidate-54: base, r1, r1, base. Two boots an arm.
W=$HOME/spark-lab/agents/candidate-56-rework-957p1; OM=$W/bin/om.sh; . $W/bin/env.sh
for spec in "base 1" "r1 1" "r1 2" "base 2"; do set -- $spec
  MINMEM_GIB=95 $OM mark-$1-boot$2 $1 "${RECIPE_ENV[@]}" "${HUB[@]}" -- \
    python3 /h/step957.py --out /out/json/mark-$1-boot$2.json
done
