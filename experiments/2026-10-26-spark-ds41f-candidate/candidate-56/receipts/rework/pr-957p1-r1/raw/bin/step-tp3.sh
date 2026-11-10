#!/bin/bash
# 0795-r1 at TP3 rank shapes on one GPU (step957.py --tp-shape 3: 24 heads, 3 output groups,
# 768-wide experts, the compressor built as TP3's rank 0), ABBA against candidate-54.
# 0795 replicates at TP3 too (COMP 2.95 -> 10.5 MB without the fix).
W=$HOME/spark-lab/agents/candidate-56-rework-957p1; OM=$W/bin/om.sh; . $W/bin/env.sh
S=1x2,1x3,1x4,1x6,2x4,2x6,3x6,4x8,4x12,4x24,8x16
for spec in "base base 1" "r1 r1 1" "r1 r1 2" "base base 2"; do set -- $spec
  MINMEM_GIB=95 $OM tp3-$1-boot$3 $2 "${RECIPE_ENV[@]}" "${HUB[@]}" -- \
    python3 /h/step957.py --tp-shape 3 --shapes $S --out /out/json/tp3-$1-boot$3.json
done
