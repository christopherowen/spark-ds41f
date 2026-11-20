#!/bin/bash
# The grid-fidelity runs on sp8, in order. Each run takes /tmp/spark-lab-gpu.lock on its own
# (gf.sh), so other jobs can run between them.
B=$HOME/spark-lab/agents/grid-fidelity; GF=$B/bin/gf.sh; . $B/bin/env.sh; mkdir -p $B/out/json $B/out/nsys
run() { local label=$1; MINMEM_GIB=95 $GF "$@"; echo "$(date -u +%FT%TZ) $label rc=$?" >> $B/out/progress.log; }
SNAP=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
# 1. Routes census of one grid step per shape and input (eager).
run census patched "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/census.py --out /out/json/census.json
# 2. Step time, ABBA, two boots an input.
for spec in "grid 1" "sample 1" "sample 2" "grid 2"; do set -- $spec
  run step-$1-boot$2 patched "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/gridstep.py --inputs $1 \
    --out /out/json/step-$1-boot$2.json
done
# 3. Kernel split: one nsys boot an input, graph nodes traced, one NVTX range a shape.
for i in grid sample; do
  run nsys-$i patched "${RECIPE_ENV[@]}" "${HUB[@]}" "${NSYS[@]}" -- /opt/nsys/bin/nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop --export=sqlite --force-overwrite=true --output /out/nsys/nsys-$i \
    python3 /h/gridstep.py --inputs $i --replays 3 --profile --out /out/json/nsys-$i.json
done
# 4. Control: the plain image's grid (the patch without token ids changes nothing).
run step-grid-base base "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/gridstep.py --inputs grid \
  --out /out/json/step-grid-base.json
echo "$(date -u +%FT%TZ) gpu runs done" >> $B/out/progress.log
