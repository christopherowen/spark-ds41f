#!/bin/bash
# run_routes.sh again after its step-routes-boot1 failed (the routes buffer, made inside the
# engine's inference mode, was filled outside it). grid-boot3 stands; then routes x2, grid-boot4, nsys.
B=$HOME/spark-lab/agents/grid-fidelity; GF=$B/bin/gf.sh; . $B/bin/env.sh; mkdir -p $B/out/json $B/out/nsys
run() { local label=$1 rc; MINMEM_GIB=95 $GF "$@"; rc=$?; echo "$(date -u +%FT%TZ) $label rc=$rc" >> $B/out/progress.log; }
# (run_routes.sh waited for ref-routes here and built the table; its step-routes-boot1 failed)
for spec in "routes 1" "routes 2" "grid 4"; do set -- $spec
  if [ $1 = routes ]; then extra=(--inputs sample --routes /out/routes-table.json); else extra=(--inputs grid); fi
  run step-$1-boot$2 patched "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/gridstep.py "${extra[@]}" \
    --out /out/json/step-$1-boot$2.json
done
run nsys-routes patched "${RECIPE_ENV[@]}" "${HUB[@]}" "${NSYS[@]}" -- /opt/nsys/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --export=sqlite --force-overwrite=true --output /out/nsys/nsys-routes \
  python3 /h/gridstep.py --inputs sample --routes /out/routes-table.json --replays 3 --profile \
  --out /out/json/nsys-routes.json
echo "$(date -u +%FT%TZ) routes runs done" >> $B/out/progress.log
