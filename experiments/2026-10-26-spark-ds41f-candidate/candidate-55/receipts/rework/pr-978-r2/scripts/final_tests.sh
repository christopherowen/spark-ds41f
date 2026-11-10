#!/usr/bin/env bash
# Recorded test runs for #978 round 2 on sp8. Each hold takes the GPU lock on its own
# (short holds) after checking that ~/spark-hold.json is absent; every run uses
# spark-ds41f:candidate-54 with an arm's files mounted (scripts/run_tests.sh), --gpus all.
#   none           candidate-54 as built
#   r1             candidate-55-r1's changed files (0792, round 1's 0793, 0794)
#   r2             candidate-55-r2's changed files (0792, round 2's 0793, 0794)
#   r1-newtests    r1 plus round 2's three test files (which new test catches which risk);
#                  their imports of round 2's new names fall back on round 1 (a shim in
#                  this diagnostic arm only)
# ARM, NEWTESTS and SUFFIX select the arms and the output labels (the final candidate:
# ARM=r2f NEWTESTS=r1-newtests-f SUFFIX=-final).
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r2
cd "$WD"
LOCK=/tmp/spark-lab-gpu.lock
ARM=${ARM:-r2}; NEWTESTS=${NEWTESTS:-r1-newtests}; SUFFIX=${SUFFIX:-}
log=$WD/raw/final-tests$SUFFIX.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }

GATE=(tests/v1/engine/test_request_submission_cleanup.py tests/entrypoints/serve/utils/test_api_cancellation.py)
K='not (test_ipc_enable_disable_consistency or test_cache_eviction_lru_cache or test_cache_eviction_shm_cache or test_processor_cache_shared_across_loras or test_sleep_wake_preserves_mm_cache_consistency)'
PR978=(tests/multimodal/test_prefix_cache_fastpath.py tests/v1/core/test_stripped_mm_scheduling.py
       tests/multimodal/test_hasher.py tests/multimodal/media/test_image.py tests/multimodal/test_cache.py
       tests/models/test_deepseek_v4_1_image_span.py)
PR978_C54=(tests/multimodal/test_hasher.py tests/multimodal/media/test_image.py tests/multimodal/test_cache.py
           tests/models/test_deepseek_v4_1_image_span.py)
DIRS=(tests/v1/engine tests/entrypoints/serve/utils)
EXTRA=(tests/v1/core/test_scheduler.py tests/v1/core/test_encoder_cache_manager.py
       tests/v1/core/test_async_scheduler.py tests/multimodal/test_processing.py
       tests/multimodal/test_registry.py tests/multimodal/test_inputs.py tests/multimodal/test_image.py
       tests/renderers/test_warmup.py tests/renderers/test_multimodal_hashes.py)

hold() {  # hold <name> <command...>: one lock hold
  [[ -e $HOME/spark-hold.json ]] && { note "hold file present; $1 not run"; return 1; }
  note "$1 waiting for the lock"
  flock "$LOCK" bash -c "[[ ! -e $HOME/spark-hold.json ]] && { ${*:2}; }" >> "$log" 2>&1
  note "$1 done"
}
T=scripts/run_tests.sh

hold gate "$T arms/$ARM gate-r2$SUFFIX --gpu -- ${GATE[*]}; $T arms/$NEWTESTS gate-r1-newtests$SUFFIX --gpu -- ${GATE[*]}"
hold pr978 "$T arms/$ARM pr978-r2-boot1$SUFFIX --gpu -- -k \"$K\" ${PR978[*]}; $T arms/$ARM pr978-r2-boot2$SUFFIX --gpu -- -k \"$K\" ${PR978[*]}; $T arms/$NEWTESTS pr978-r1-newtests$SUFFIX --gpu -- -k \"$K\" ${PR978[*]}; $T arms/none pr978-c54$SUFFIX --gpu -- -k \"$K\" ${PR978_C54[*]}"
hold dirs "timeout 3600 $T arms/$ARM dirs-r2$SUFFIX --gpu -- --continue-on-collection-errors ${DIRS[*]}"
hold extra "timeout 3600 $T arms/none extra-c54$SUFFIX --gpu -- --continue-on-collection-errors ${EXTRA[*]}; timeout 3600 $T arms/$ARM extra-r2$SUFFIX --gpu -- --continue-on-collection-errors ${EXTRA[*]}"
note "all done"
