#!/bin/bash
# #957 part 1 (0795): the decode step at TP4 rank shapes on one GPU (step957.py), ABBA.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-957p1/json
for spec in "base 1" "c56p1 1" "c56p1 2" "base 2"; do set -- $spec
  MINMEM_GIB=95 $OM 957p1 step-$1-boot$2 $1 "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/step957.py --out /out/json/step-$1-boot$2.json
done
