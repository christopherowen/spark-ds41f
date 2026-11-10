#!/bin/bash
# #957 part 2 (0797): sparse-MLA cells at TP3/TP4 heads (h957p2.py), ABBA, then the PR's test.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-957p2/json
for spec in "base 1" "c57p2 1" "c57p2 2" "base 2"; do set -- $spec
  $OM 957p2 mla-$1-boot$2 $1 "${RECIPE_ENV[@]}" -- python3 /h/h957p2.py --boot $2 --out /out/json/mla-$1-boot$2.json
done
for k in 1 2; do
  $OM 957p2 pytest-c57p2-boot$k c57p2 -- pytest --noconftest -o cache_dir=/tmp/pc -q tests/kernels/attention/test_deepseek_v41_tilelang.py -k test_sparse_mla_plan_routes
done
