#!/usr/bin/env bash
# Recorded test runs for #978 round 3 on sp8. Each hold takes the GPU lock on its own
# (short holds) after checking that ~/spark-hold.json is absent; every run uses
# spark-ds41f:candidate-54 with an arm's files mounted (scripts/run_tests.sh, which does
# not take the lock), --gpus all.
#   r3            candidate-55-r3's 27 changed files (0792, this 0793, 0794)
#   r2-newtest    round 2's 26 files plus round 3's new test file
#   c54-newtest   candidate-54 as built plus round 3's new test file
#   *-async       the arm plus a dev-only wrapper running test_admission_control.py's async
#                 test with asyncio.run (the environment has no async plugin)
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r3
cd "$WD"
LOCK=/tmp/spark-lab-gpu.lock
log=$WD/raw/tests/final-tests.log
mkdir -p raw/tests
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }

GATE=(tests/v1/engine/test_request_submission_cleanup.py tests/entrypoints/serve/utils/test_api_cancellation.py)
K='not (test_ipc_enable_disable_consistency or test_cache_eviction_lru_cache or test_cache_eviction_shm_cache or test_processor_cache_shared_across_loras or test_sleep_wake_preserves_mm_cache_consistency)'
PR978=(tests/multimodal/test_prefix_cache_fastpath.py tests/v1/core/test_stripped_mm_scheduling.py
       tests/v1/core/test_text_request_scheduling.py
       tests/multimodal/test_hasher.py tests/multimodal/media/test_image.py tests/multimodal/test_cache.py
       tests/models/test_deepseek_v4_1_image_span.py)
NEW=tests/v1/core/test_text_request_scheduling.py
DIRS=(tests/v1/engine tests/entrypoints/serve/utils)
EXTRA=(tests/v1/core/test_scheduler.py tests/v1/core/test_encoder_cache_manager.py
       tests/v1/core/test_async_scheduler.py tests/multimodal/test_processing.py
       tests/multimodal/test_registry.py tests/multimodal/test_inputs.py tests/multimodal/test_image.py
       tests/renderers/test_warmup.py tests/renderers/test_multimodal_hashes.py
       tests/v1/core/test_sink_draft_zero.py tests/v1/core/test_prefix_caching.py)

hold() {  # hold <name> <command...>: one lock hold
  [[ -e $HOME/spark-hold.json ]] && { note "hold file present; $1 not run"; return 1; }
  note "$1 waiting for the lock"
  flock "$LOCK" bash -c "[[ ! -e $HOME/spark-hold.json ]] && { ${*:2}; }" >> "$log" 2>&1
  note "$1 done"
}
T=scripts/run_tests.sh

hold new "$T arms/c54-newtest tests/new-c54 --gpu -- -k 'not stripped_item_miss' $NEW; $T arms/r2-newtest tests/new-r2 --gpu -- $NEW; $T arms/r3 tests/new-r3 --gpu -- $NEW"
A=tests/v1/engine/test_zz_hidden_async.py
hold async "$T arms/c54-async tests/async-c54 --gpu -- $A; $T arms/r2-async tests/async-r2 --gpu -- $A; $T arms/r3-async tests/async-r3 --gpu -- $A"
hold gate "$T arms/r3 tests/gate-r3 --gpu -- ${GATE[*]}"
hold pr978 "$T arms/r3 tests/pr978-r3-boot1 --gpu -- -k \"$K\" ${PR978[*]}; $T arms/r3 tests/pr978-r3-boot2 --gpu -- -k \"$K\" ${PR978[*]}"
hold dirs "timeout 3600 $T arms/r3 tests/dirs-r3 --gpu -- --continue-on-collection-errors ${DIRS[*]}"
hold extra "timeout 3600 $T arms/none tests/extra-c54 --gpu -- --continue-on-collection-errors ${EXTRA[*]}; timeout 3600 $T arms/r3 tests/extra-r3 --gpu -- --continue-on-collection-errors ${EXTRA[*]}"
note "all done"
