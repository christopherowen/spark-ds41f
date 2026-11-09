#!/bin/bash
# #961: the PR's bench_l2_prefetch_ab.py at 9362f944, unchanged, behind a >=250 ms GPU
# warm-up (h/warm.py), with the recipe's VLLM_L2_PREFETCH_GRID=2 and
# VLLM_L2_PREFETCH_KERNELS=tilelang, ABBA in one session, then the PR's tests on c51 (with the release bundle's tblib stand-in).
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh; mkdir -p $B/out/pr-961/json
S=/src/pr-961/experiments/2026-11-08-l2-prefetch-pacing/bench_l2_prefetch_ab.py
for spec in "base 1" "c51 1" "c51 2" "base 2"; do set -- $spec
  $OM 961 l2pf-$1-boot$2 $1 -e VLLM_L2_PREFETCH_GRID=2 -e VLLM_L2_PREFETCH_KERNELS=tilelang -- \
    python3 /h/warm.py $S --boot $2 --replays 35 --out /out/json/l2pf-$1-boot$2.json
done
for k in 1 2; do
  $OM 961 pytest-c51-boot$k c51 -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h -- pytest -p tblib_stub -v \
    tests/models/kimi_k3/test_l2_prefetch.py \
    tests/models/test_deepseek_v4_1_l2_schedule.py \
    tests/models/test_glm5next_l2_prefetch_persist.py \
    tests/models/test_glm5next_tilelang_l2_prefetch.py
done
