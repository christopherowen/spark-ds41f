#!/usr/bin/env bash
# CPU tests for the late-admission prefill fix (0226), in throwaway containers from the
# candidate image vllm-ds41f-kkref:19f2c20ed4d6-1022-2 on dgx4 (idle, not in the serving
# set). The image's vLLM tree (c282fa51) is the fix's base; the fix's changed files are
# mounted over it. No network.
#
#   run-tests.sh cpu [TREE]       the late-admission, combined and sink suites (CPU), and
#                                 the scheduler suites (GPU container), with the fix mounted
#   run-tests.sh baseline         the scheduler suites without the fix (the image as built)
#
# TREE is the prepared vLLM tree on branch late-admission-prefill (default: the lab's).
# --confcutdir skips the root tests/conftest.py (tblib is not in the image). The scheduler
# suites build a ModelConfig for facebook/opt-125m, mounted read-only from the
# sink-rejection runs' cache.
set -u
TREE=${2:-$HOME/spark-lab/la-prefill/repo/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-16175289/src/vllm}
HF_CACHE=${HF_CACHE:-$HOME/spark-lab/sink-rejection/hf-cache}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-1022-2
FILES=(vllm/v1/engine/core.py vllm/v1/core/sched/interface.py vllm/v1/core/sched/scheduler.py
       tests/v1/engine/test_late_admission.py)

run() {
  local gpu=$1 mount=$2; shift 2
  local args=(--rm --network none --memory 48g -w "$V" -e HF_HUB_OFFLINE=1
              -v "$HF_CACHE:/root/.cache/huggingface:ro")
  [ "$gpu" = gpu ] && args+=(--gpus all --ipc host)
  if [ "$mount" = fix ]; then
    local f
    for f in "${FILES[@]}"; do args+=(-v "$TREE/$f:$V/$f:ro"); done
  fi
  docker run "${args[@]}" --entrypoint python3 "$IMG" \
    -m pytest -q -p no:cacheprovider --confcutdir=tests/v1 "$@"
}

LATE=(tests/v1/engine/test_late_admission.py tests/v1/engine/test_late_admission_sink.py)
SINK=(tests/v1/core/test_sink_draft_zero.py)
SCHEDULER=(tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py)

case ${1:-} in
  cpu)
    echo "== late admission, engine side, and with sink rejection through the engine-core loop"
    run cpu fix "${LATE[@]}"
    echo "== sink rejection: scheduler, frontend, default"
    run cpu fix "${SINK[@]}"
    echo "== scheduler suites (GPU container), with the fix"
    run gpu fix -rf "${SCHEDULER[@]}"
    ;;
  baseline)
    echo "== scheduler suites (GPU container), the image as built"
    run gpu image -rf "${SCHEDULER[@]}"
    ;;
  *) echo "usage: $0 cpu [TREE]|baseline" >&2; exit 2 ;;
esac
