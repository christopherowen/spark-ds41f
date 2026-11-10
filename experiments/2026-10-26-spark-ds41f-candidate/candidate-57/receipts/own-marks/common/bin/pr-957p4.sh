#!/bin/bash
# #957 part 4 (0798): confidence cells (h957p4.py), ABBA, then the PR's test.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-957p4/json
for spec in "base 1" "c57p4 1" "c57p4 2" "base 2"; do set -- $spec
  $OM 957p4 conf-$1-boot$2 $1 "${RECIPE_ENV[@]}" -- python3 /h/h957p4.py --arm $1 --boot $2 --out /out/json/conf-$1-boot$2.json
done
for k in 1 2; do
  $OM 957p4 pytest-c57p4-boot$k c57p4 -- pytest --noconftest -o cache_dir=/tmp/pc -q tests/kernels/test_deepseek_v41_tilelang_confidence.py
done
