#!/usr/bin/env bash
# #978 round 2's recorded measurements of the final candidate (arms/r2) on sp8, in order.
# Each serving_session.sh call is one short hold of /tmp/spark-lab-gpu.lock (taken there,
# after checking ~/spark-hold.json).
#   (ARM and PREFIX select the candidate arm and the output directories; the final
#   candidate: ARM=r2f PREFIX=final2)
#   final-mark     round 1's real-AsyncLLM frontend bench (exp/serving_path_bench.py, unchanged),
#                  ABBA against candidate-54, two boots per arm: base r2 r2 base
#   final-text     text-only requests through the same frontend (exp/serving_text_bench.py),
#                  ABBA, two boots per arm
#   final-prodenv  the mark's bench with the recipe's process environment for the frontend
#                  (config/cluster.json: MALLOC_ARENA_MAX, MALLOC_MMAP_THRESHOLD_,
#                  OMP_NUM_THREADS, VLLM_MAX_IMAGE_PIXELS), ABBA, two boots per arm
#   final-diag     the stage diagnostic (exp/serving_stage_diag.py): base r1 r2 r2 r1 base
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r2
cd "$WD"
S=scripts/serving_session.sh
ARM=${ARM:-r2}; P=${PREFIX:-final}
ENV=MALLOC_ARENA_MAX=2,MALLOC_MMAP_THRESHOLD_=1048576,OMP_NUM_THREADS=8,VLLM_MAX_IMAGE_PIXELS=16777216
$S $P-mark serving-base-boot1:none serving-r2-boot1:$ARM serving-r2-boot2:$ARM serving-base-boot2:none
BENCH=serving_text_bench.py $S $P-text text-base-boot1:none text-r2-boot1:$ARM text-r2-boot2:$ARM text-base-boot2:none
$S $P-prodenv prodenv-base-boot1:none:$ENV prodenv-r2-boot1:$ARM:$ENV prodenv-r2-boot2:$ARM:$ENV prodenv-base-boot2:none:$ENV
BENCH=serving_stage_diag.py $S $P-diag diag-base-1:none diag-r1-1:r1 diag-r2-1:$ARM diag-r2-2:$ARM diag-r1-2:r1 diag-base-2:none
echo "$(date -u +%FT%TZ) measure done" >> raw/measure.log
