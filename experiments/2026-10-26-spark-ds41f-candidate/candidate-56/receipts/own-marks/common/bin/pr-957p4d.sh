#!/bin/bash
# #957 part 4 (0798): the dummy run's DSpark draft dispatch at TP3 rank shapes (step957.py
# --tp-shape 3 --log-draft-dispatch): which draft graph the (4, 24) step takes.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh
for arm in base c57p4; do
  MINMEM_GIB=95 $OM 957p4 dispatch-tp3-$arm $arm "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/step957.py --tp-shape 3 \
    --log-draft-dispatch --shapes 1x6,4x8,4x20,4x24,8x16 --replays 6 --out /out/json/dispatch-tp3-$arm.json
done
