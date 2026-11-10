#!/usr/bin/env bash
# Recorded test runs for the #978 round-1 rework on sp8. Each step takes the
# GPU lock on its own (short holds); every run uses spark-ds41f:candidate-54
# with an arm's files mounted (scripts/run_tests.sh), --gpus all.
#   arm-c55          candidate-55's changed files (410463b3)
#   arm-r1           candidate-55-r1's changed files (corrected 0793, then 0794)
#   arm-c55-newtests candidate-55's files plus r1's two changed test files
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978
cd "$WD"
LOCK=/tmp/spark-lab-gpu.lock
log=$WD/raw/final-tests.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
[[ -e $HOME/spark-hold.json ]] && { note "hold file present; not running"; exit 1; }

GATE=(tests/v1/engine/test_request_submission_cleanup.py tests/entrypoints/serve/utils/test_api_cancellation.py)
PR978=(-k 'not (test_ipc_enable_disable_consistency or test_cache_eviction_lru_cache or test_cache_eviction_shm_cache or test_processor_cache_shared_across_loras or test_sleep_wake_preserves_mm_cache_consistency)'
       tests/multimodal/test_prefix_cache_fastpath.py tests/multimodal/test_hasher.py
       tests/multimodal/media/test_image.py tests/multimodal/test_cache.py
       tests/models/test_deepseek_v4_1_image_span.py)
DIRS=(tests/v1/engine tests/entrypoints/serve/utils)

step1() {
  scripts/run_tests.sh arm-c55 gate-c55 --gpu -- "${GATE[@]}"
  scripts/run_tests.sh arm-r1 gate-r1 --gpu -- "${GATE[@]}"
  scripts/run_tests.sh arm-c55-newtests gate-c55-newtests --gpu -- "${GATE[@]}"
  scripts/run_tests.sh arm-c55 pr978-c55 --gpu -- "${PR978[@]}"
  scripts/run_tests.sh arm-r1 pr978-r1-boot1 --gpu -- "${PR978[@]}"
  scripts/run_tests.sh arm-r1 pr978-r1-boot2 --gpu -- "${PR978[@]}"
  scripts/run_tests.sh arm-c55-newtests pr978-c55-newtests --gpu -- "${PR978[@]}"
}
# One hold for the gate files, #978's pytest and the own mark's session.
note "hold 1 (gate files, #978's pytest, own mark) waiting for the lock"
flock "$LOCK" bash -c "$(declare -p GATE PR978); $(declare -f step1); cd $WD; step1; scripts/own_mark.sh" >> "$log" 2>&1
note "hold 1 done"

for arm in arm-c55 arm-r1; do
  note "dirs on $arm waiting for the lock"
  flock "$LOCK" timeout 3600 scripts/run_tests.sh "$arm" "dirs-${arm#arm-}" --gpu -- "${DIRS[@]}" >> "$log" 2>&1
  note "dirs on $arm done"
done
note "all done"
