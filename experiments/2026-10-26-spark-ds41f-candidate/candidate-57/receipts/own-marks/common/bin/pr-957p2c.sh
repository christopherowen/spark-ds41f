#!/bin/bash
# #957 part 2 (0797) confirmation (step 8, "Confirm in one session first"): 4+4 boots, ABBA ABBA,
# the mark's TP3 12/16-row cells, TP4's 6- and 32-row cells (the two that read slower) and
# TP4's 12/16-row cells, each at both index-length mixes. Same harness (h957p2.py with --only),
# same image and arms as the first run.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-957p2c/json
ONLY="tp3:12,16;tp4:6,12,16,32"
for spec in "base 1" "c57p2 1" "c57p2 2" "base 2" "base 3" "c57p2 3" "c57p2 4" "base 4"; do set -- $spec
  $OM 957p2c mlac-$1-boot$2 $1 "${RECIPE_ENV[@]}" -- python3 /h/h957p2.py --boot $2 --only "$ONLY" --out /out/json/mlac-$1-boot$2.json
done
