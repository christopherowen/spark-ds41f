#!/bin/bash
# queue.sh: every own-mark run of candidate-52, in order. Each PR: queue A (the PR's own bench, as its
# README gives it) ABBA = base boot1, c52 boot1, c52 boot2, base boot2; then queue B (our harness) ABBA in
# the same order; then queue C (the PR's tests on the candidate arm, two boots). A "boot" is a fresh
# container. Each section runs as `OM_LOCK_HELD=1 flock /tmp/spark-lab-gpu.lock bash queue.sh <section>`
# (run_all.sh), so one PR's queue runs back to back inside one lock hold; om.sh records host state per run.
B=$HOME/spark-lab/agents/candidate-52-own-marks
OM="bash $B/om.sh"
ABBA="base:1 c52:1 c52:2 base:2"
which=${1:-all}

# ---- overlay check (verify_overlay2.py: verify_overlay.py with closure names counted too) ------------
if [ $which = all ] || [ $which = common ]; then
  V="-v $B/verify_overlay2.py:/om/verify_overlay.py:ro -v $B/changed-files.txt:/om/changed-files.txt:ro"
  $OM common verify2-base base $V -- python3 /om/verify_overlay.py base
  $OM common verify2-c52 c52 $V -- python3 /om/verify_overlay.py c52
fi

# ---- #1030 (0781) ---------------------------------------------------------------------------------
E1030=$B/src/pr-1030/experiments/2026-11-09-deepseek-reference-alignment
M1030="-v $E1030:/workspace/exp:ro -v $E1030/bench_reference_alignment.py:/bench/bench_reference_alignment.py:ro -v $B/ovl/c52/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py:/ref/test_deepseek_v41_tilelang_moe.py:ro -v $B/h:/h:ro"
if [ $which = all ] || [ $which = 1030 ]; then
  for ab in $ABBA; do a=${ab%:*}; k=${ab#*:}
    $OM 1030 bench-$a-boot$k $a $M1030 -- python3 /workspace/exp/bench_reference_alignment.py /out/bench-$a-boot$k.json; done
  for ab in $ABBA; do a=${ab%:*}; k=${ab#*:}
    $OM 1030 h1030-$a-boot$k $a $M1030 -- python3 /h/h1030.py $a-boot$k; done
  for k in 1 2; do
    $OM 1030 pytest-c52-boot$k c52 -- python3 -m pytest --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources \
      tests/kernels/moe/test_deepseek_v41_tilelang_moe.py tests/kernels/test_deepseek_v41_tilelang_activation.py; done
fi

# ---- #968 (0782) ----------------------------------------------------------------------------------
E968=$B/src/pr-968/experiments/2026-11-08-remaining-token-draft-cap
M968="-e PYTHONPATH=/opt/spark-ds41f/candidate/vllm -v $E968/screen_remaining_token_cap.py:/tmp/screen_remaining_token_cap.py:ro -v $B/h:/h:ro"
if [ $which = all ] || [ $which = 968 ]; then
  for ab in $ABBA; do a=${ab%:*}; k=${ab#*:}
    $OM 968 screen-$a-boot$k $a $M968 -- python3 /tmp/screen_remaining_token_cap.py; done
  for ab in $ABBA; do a=${ab%:*}; k=${ab#*:}
    $OM 968 h968-$a-boot$k $a $M968 -- python3 /h/h968.py $a-boot$k; done
  for k in 1 2; do
    $OM 968 pytest-c52-boot$k c52 -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm -- python3 -m pytest --noconftest -o addopts= -q \
      tests/v1/core/test_copy_drafts_scheduler.py tests/v1/core/test_sink_draft_zero.py \
      tests/v1/spec_decode/test_adaptive_verification.py tests/v1/spec_decode/test_acceptance_estimator.py; done
fi

# ---- #969 (0783) ----------------------------------------------------------------------------------
mkdir -p $B/nvme
M969="--cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --security-opt=seccomp=unconfined --security-opt=label=disable -v $B/nvme:/mnt/nvme -v $B/bench969:/bench969:ro -e ENGRAM_ROWS_TEST_DIR=/mnt/nvme/engram-rows-test"
if [ $which = all ] || [ $which = 969 ]; then
  for ab in $ABBA; do a=${ab%:*}; k=${ab#*:}
    $OM 969 bench-cache-$a-boot$k $a $M969 -- python3 /bench969/run969.py $a-boot$k 20; done
  for ab in $ABBA; do a=${ab%:*}; k=${ab#*:}
    $OM 969 rows-$a-boot$k $a $M969 -- python3 /bench969/rows969.py $a-boot$k; done
  for k in 1 2; do
    $OM 969 pytest-c52-boot$k c52 $M969 -- python3 -m pytest --noconftest -v tests/kernels/test_deepseek_v41_engram_rows.py; done
fi
echo "queue $which done $(date -u +%FT%TZ)"
