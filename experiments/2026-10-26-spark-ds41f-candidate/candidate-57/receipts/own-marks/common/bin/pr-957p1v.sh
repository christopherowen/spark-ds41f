#!/bin/bash
# #957 part 1 (0795): mechanism controls, one variable each: the L2 prefetch off
# (VLLM_DS41_L2_PREFETCH=0), and only the COMP window dropped (--no-comp-window).
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh
S=1x2,1x3,1x6,4x8,16x32
for spec in "base 1" "c56p1 1" "c56p1 2" "base 2"; do set -- $spec
  MINMEM_GIB=95 $OM 957p1 nopf-$1-boot$2 $1 "${RECIPE_ENV[@]}" -e VLLM_DS41_L2_PREFETCH=0 "${HUB[@]}" -- python3 /h/step957.py --shapes $S --out /out/json/nopf-$1-boot$2.json
done
for spec in "base 1" "c56p1 1" "c56p1 2" "base 2"; do set -- $spec
  MINMEM_GIB=95 $OM 957p1 nocomp-$1-boot$2 $1 "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/step957.py --shapes $S --no-comp-window --out /out/json/nocomp-$1-boot$2.json
done
