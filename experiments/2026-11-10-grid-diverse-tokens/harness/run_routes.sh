#!/bin/bash
# After run_ref.sh: real text's expert ids (the reference forward's) injected into the sample's
# rows, ABBA against the grid's own inputs in the same session, then one nsys boot.
B=$HOME/spark-lab/agents/grid-fidelity; GF=$B/bin/gf.sh; . $B/bin/env.sh; mkdir -p $B/out/json $B/out/nsys
run() { local label=$1 rc; MINMEM_GIB=95 $GF "$@"; rc=$?; echo "$(date -u +%FT%TZ) $label rc=$rc" >> $B/out/progress.log; }
until grep -q "ref-routes rc" $B/out/progress.log 2>/dev/null; do sleep 30; done
docker run --rm --network none -v $B/out:/out -v $B/h:/h:ro -v $B/sb:/sb:ro --entrypoint python3 \
  spark-ds41f:candidate-57 /h/routes_table.py /out/ref-routes.pt /out/routes-table.json >> $B/out/progress.log 2>&1
for spec in "grid 3" "routes 1" "routes 2" "grid 4"; do set -- $spec
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
