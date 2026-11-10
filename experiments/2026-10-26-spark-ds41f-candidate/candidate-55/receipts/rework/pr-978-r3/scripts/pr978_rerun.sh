#!/usr/bin/env bash
# #978's pytest set on r3 again, two boots, after final_tests.sh's pr978 hold errored in
# setup: its labels tests/<name> made run_tests.sh's --basetemp /c/pytest-tmp-tests/<name>,
# whose parent did not exist, so the 4 tmp_path tests of media/test_image.py errored
# (raw/tests/pr978-r3-boot{1,2}.txt, kept). The parent now exists. One lock hold.
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r3
cd "$WD"
K='not (test_ipc_enable_disable_consistency or test_cache_eviction_lru_cache or test_cache_eviction_shm_cache or test_processor_cache_shared_across_loras or test_sleep_wake_preserves_mm_cache_consistency)'
PR978="tests/multimodal/test_prefix_cache_fastpath.py tests/v1/core/test_stripped_mm_scheduling.py tests/v1/core/test_text_request_scheduling.py tests/multimodal/test_hasher.py tests/multimodal/media/test_image.py tests/multimodal/test_cache.py tests/models/test_deepseek_v4_1_image_span.py"
[[ -e $HOME/spark-hold.json ]] && { echo "hold file present"; exit 1; }
flock /tmp/spark-lab-gpu.lock bash -c "[[ ! -e $HOME/spark-hold.json ]] && scripts/run_tests.sh arms/r3 tests/pr978-r3-boot1-rerun --gpu -- -k \"$K\" $PR978 && scripts/run_tests.sh arms/r3 tests/pr978-r3-boot2-rerun --gpu -- -k \"$K\" $PR978" >> raw/tests/pr978-rerun.log 2>&1
echo "$(date -u +%FT%TZ) pr978 rerun done" >> raw/tests/pr978-rerun.log
