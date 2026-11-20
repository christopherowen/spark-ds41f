#!/bin/bash
# After run_routes.sh: the patch tests again, with the tblib stand-in the serving image needs
# for tests/conftest.py (the first attempt, pytest-patched, stopped at the conftest import).
B=$HOME/spark-lab/agents/grid-fidelity; GF=$B/bin/gf.sh
until grep -q "routes runs done" $B/out/progress.log 2>/dev/null; do sleep 30; done
MINMEM_GIB=95 $GF pytest-patched-r2 patched -- python3 -m pytest -p tblib_stub -o cache_dir=/tmp/pc -q \
  tests/v1/worker/test_engine_grid_token_ids.py tests/v1/worker/test_gpu_input_batch_v2.py tests/v1/worker/test_dead_rows.py
rc=$?; echo "$(date -u +%FT%TZ) pytest-patched-r2 rc=$rc" >> $B/out/progress.log
echo "$(date -u +%FT%TZ) tail runs done" >> $B/out/progress.log
