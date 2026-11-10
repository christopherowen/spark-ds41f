#!/bin/bash
# #988 dead-row part (0794): sampler cells (h988.py), ABBA, then the PR's tests.
B=$HOME/spark-lab/agents/round3-own-marks; OM=$B/bin/om.sh; . $B/bin/env.sh; mkdir -p $B/out/pr-988/json
for spec in "base 1" "c55dr 1" "c55dr 2" "base 2"; do set -- $spec
  $OM 988 sampler-$1-boot$2 $1 "${RECIPE_ENV[@]}" -- python3 /h/h988.py --arm $1 --boot $2 --out /out/json/sampler-$1-boot$2.json
done
for k in 1 2; do
  $OM 988 pytest-c55dr-boot$k c55dr -- pytest --noconftest -o cache_dir=/tmp/pc -q tests/v1/worker/test_dead_rows.py \
    tests/v1/worker/test_gpu_sampling_states_gather.py tests/kernels/test_deepseek_v41_copy_logits.py tests/v1/worker/test_prepare_kernels.py
done
