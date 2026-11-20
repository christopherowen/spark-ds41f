#!/bin/bash
# After run_all.sh: the routes census again (eager needs adaptive verification off), and the
# patch's vLLM tests. Each run takes /tmp/spark-lab-gpu.lock on its own (gf.sh).
B=$HOME/spark-lab/agents/grid-fidelity; GF=$B/bin/gf.sh; . $B/bin/env.sh; mkdir -p $B/out/json
run() { local label=$1 rc; MINMEM_GIB=95 $GF "$@"; rc=$?; echo "$(date -u +%FT%TZ) $label rc=$rc" >> $B/out/progress.log; }
until grep -q "gpu runs done" $B/out/progress.log 2>/dev/null; do sleep 30; done
run census-r2 patched "${RECIPE_ENV[@]}" "${HUB[@]}" -- python3 /h/census.py --out /out/json/census.json
run pytest-patched patched -- python3 -m pytest -q tests/v1/worker/test_engine_grid_token_ids.py \
  tests/v1/worker/test_gpu_input_batch_v2.py tests/v1/worker/test_dead_rows.py
echo "$(date -u +%FT%TZ) more runs done" >> $B/out/progress.log
