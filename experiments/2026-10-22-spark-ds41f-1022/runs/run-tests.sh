#!/usr/bin/env bash
# Unit tests for the spark-ds41f-1022 candidate's vLLM series (r6x's 7316c7f9f + 0185, 0221,
# 0222, 0223), in throwaway containers from the promoted r6x image on dgx4 (idle, not in the
# serving set), with the candidate's changed files mounted over the image's sources (the
# image's vLLM tree is the series' base, tree 4d6851c8). No network; dgx4's own GPU.
#
#   run-tests.sh sync [vllm-worktree]   (on the Mac) copies the files that differ from
#                                       7316c7f9f into dgx4:$LAB/candidate
#   run-tests.sh cpu                    each piece's CPU suites and the combined ones
#   run-tests.sh gpu                    the suites that need the GPU, and the existing
#                                       scheduler and batch-queue suites
#   run-tests.sh pytest cpu|gpu ARGS    pytest ARGS alone (a rerun of one suite)
#
# --confcutdir skips the root tests/conftest.py (tblib is not in the image). The scheduler
# suites build a ModelConfig for facebook/opt-125m, whose config and tokenizer files are
# mounted read-only from the sink-rejection runs' cache.
set -u
LAB=${LAB:-$HOME/spark-lab/spark-ds41f-1022}
HF_CACHE=${HF_CACHE:-$HOME/spark-lab/sink-rejection/hf-cache}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
BASE=7316c7f9f

mounts() {
  local f
  for f in $(cd "$LAB/candidate" && find . -type f -name '*.py' | sed 's#^\./##'); do
    printf -- '-v %s/candidate/%s:%s/%s:ro ' "$LAB" "$f" "$V" "$f"
  done
}

run() {
  local gpu=$1 conf=$2; shift 2
  local args=(--rm --network none --memory 48g -w "$V" -e HF_HUB_OFFLINE=1
              -v "$HF_CACHE:/root/.cache/huggingface:ro")
  [ "$gpu" = gpu ] && args+=(--gpus all --ipc host)
  docker run "${args[@]}" $(mounts) --entrypoint python3 "$IMG" \
    -m pytest -q -p no:cacheprovider --confcutdir="$conf" "$@"
}

# 0185: the TP group's broadcast on the RoCE policy.
BROADCAST=(tests/distributed/test_roce_policy_no_fallback.py)
# 0221 and 0223: sink rejection, default on for DeepSeek V4.1.
SINK_CPU=(tests/v1/core/test_sink_draft_zero.py)
SINK_GPU=(tests/v1/worker/test_gpu_sink.py)
# 0222: late admission; the worker file includes the captured-graph GPU test.
LATE=(tests/v1/engine/test_late_admission.py tests/v1/worker/test_late_admission.py)
# Late admission with sink rejection: the real engine-core loop.
COMBINED=(tests/v1/engine/test_late_admission_sink.py)
EXISTING_CPU=(tests/v1/core/test_output.py tests/v1/core/test_repetition_detection.py
              tests/v1/engine/test_output_processor.py)
EXISTING_GPU=(tests/v1/engine/test_structured_output_draft_handoff.py
              tests/v1/engine/test_compute_fairness_feedback.py
              tests/v1/worker/test_draft_tokens_handler.py
              tests/v1/worker/test_gpu_gumbel_sample.py
              tests/v1/worker/test_gpu_rejection_sampler_chunking.py
              tests/v1/worker/test_gpu_sampler_flags.py
              tests/v1/worker/test_dead_rows.py
              tests/v1/worker/test_async_verified_draft_counts.py
              tests/v1/worker/test_gpu_input_batch_v2.py
              tests/v1/spec_decode/test_dflash_prepare_inputs.py
              tests/v1/spec_decode/test_adaptive_verification.py)
SCHEDULER=(tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py)

case ${1:-} in
  sync)
    src=${2:-$HOME/projects/vllm-ds41-next}
    list=$(mktemp)
    (cd "$src" && { git diff --name-only $BASE -- vllm tests; git ls-files --others --exclude-standard vllm tests; } \
      | sort -u | while read -r f; do [ -f "$f" ] && echo "$f"; done > "$list"
     tar -C "$src" -cf - -T "$list") \
      | ssh dgx4 "rm -rf ~/spark-lab/spark-ds41f-1022/candidate && mkdir -p ~/spark-lab/spark-ds41f-1022/candidate && tar -C ~/spark-lab/spark-ds41f-1022/candidate -xf -"
    wc -l < "$list"; rm -f "$list"
    ;;
  cpu)
    echo "== 0185 broadcast on the RoCE policy (CPU)"
    run cpu tests/distributed "${BROADCAST[@]}"
    echo "== 0221/0223 sink rejection: scheduler, frontend, default (CPU)"
    run cpu tests/v1 "${SINK_CPU[@]}"
    echo "== 0222 late admission, engine side (CPU)"
    run cpu tests/v1 tests/v1/engine/test_late_admission.py
    echo "== late admission with sink rejection: engine-core loop (CPU)"
    run cpu tests/v1 "${COMBINED[@]}"
    echo "== existing CPU suites: scheduler output, repetition stop, output processor"
    run cpu tests/v1 "${EXISTING_CPU[@]}"
    ;;
  gpu)
    echo "== sink rejection: kernels, samplers and the collapse sequence (GPU)"
    run gpu tests/v1 "${SINK_GPU[@]}"
    echo "== late admission, both sides, captured graph included (GPU)"
    run gpu tests/v1 "${LATE[@]}" "${COMBINED[@]}"
    echo "== existing batch-queue, sampling, verification and drafter-input suites (GPU)"
    run gpu tests/v1 "${EXISTING_GPU[@]}"
    echo "== scheduler suites (GPU container)"
    run gpu tests/v1 "${SCHEDULER[@]}"
    ;;
  pytest)
    shift
    gpu=$1; shift
    run "$gpu" tests/v1 "$@"
    ;;
  *) echo "usage: $0 sync [vllm-worktree]|cpu|gpu|pytest cpu|gpu ARGS" >&2; exit 2 ;;
esac
