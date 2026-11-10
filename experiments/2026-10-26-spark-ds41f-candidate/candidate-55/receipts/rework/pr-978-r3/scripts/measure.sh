#!/usr/bin/env bash
# #978 round 3's serving-path measurements on sp8 (round 2's benches, unchanged), in order.
# Each serving_session.sh call is one hold of /tmp/spark-lab-gpu.lock (taken there, after
# checking ~/spark-hold.json). base = spark-ds41f:candidate-54 as built; r3 = its image
# with candidate-55-r3's 27 changed files mounted (arms/r3).
#   r3-mark     round 1's real-AsyncLLM frontend bench (exp/serving_path_bench.py), ABBA
#   r3-text     text-only requests through the same frontend (exp/serving_text_bench.py), ABBA twice
#               (four boots per arm: the short prompt's sub-millisecond medians vary 0.15-0.6 ms by boot)
#   r3-prodenv  the mark's bench with the recipe's frontend environment, ABBA
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r3
cd "$WD"
S=scripts/serving_session.sh
ENV=MALLOC_ARENA_MAX=2,MALLOC_MMAP_THRESHOLD_=1048576,OMP_NUM_THREADS=8,VLLM_MAX_IMAGE_PIXELS=16777216
$S r3-mark serving-base-boot1:none serving-r3-boot1:r3 serving-r3-boot2:r3 serving-base-boot2:none
BENCH=serving_text_bench.py $S r3-text text-base-boot1:none text-r3-boot1:r3 text-r3-boot2:r3 text-base-boot2:none \
  text-base-boot3:none text-r3-boot3:r3 text-r3-boot4:r3 text-base-boot4:none
$S r3-prodenv prodenv-base-boot1:none:$ENV prodenv-r3-boot1:r3:$ENV prodenv-r3-boot2:r3:$ENV prodenv-base-boot2:none:$ENV
echo "$(date -u +%FT%TZ) measure done" >> raw/measure.log
