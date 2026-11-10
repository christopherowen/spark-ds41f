#!/bin/bash
# #957 part 1 (0795): one nsys boot an arm (kernel timeline of the grid's 1x2 and 1x6 steps,
# graph nodes traced), after the timing boots.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-957p1/json $B/out/pr-957p1/nsys
for arm in base c56p1; do
  MINMEM_GIB=95 $OM 957p1 nsys-$arm $arm "${RECIPE_ENV[@]}" "${HUB[@]}" "${NSYS[@]}" -- /opt/nsys/bin/nsys profile \
    --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop --export=sqlite --force-overwrite=true --output /out/nsys/nsys-$arm \
    python3 /h/step957.py --out /out/json/nsys-$arm.json --shapes 1x2,1x6,16x32 --replays 3 \
    --profile-shapes 1x2,1x6 --profile-replays 4
done
