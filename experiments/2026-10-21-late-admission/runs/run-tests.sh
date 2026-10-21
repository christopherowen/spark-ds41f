#!/usr/bin/env bash
# Unit tests for vLLM 0222 (late admission), in a throwaway container from the promoted
# r6x image on an idle node, with the candidate's changed files mounted over the image's
# sources (the image's vLLM tree is the candidate's base, r6x 7316c7f9f, tree 4d6851c8).
# No network; no serving node touched.
#
#   run-tests.sh sync [node]       (on the Mac) copies overlay/ to <node>:$LAB/candidate
#   run-tests.sh cpu               the late-admission suites, CPU only
#   run-tests.sh gpu               the same plus the captured-graph test and the engine
#                                  suites that drive step_with_batch_queue, on the GPU
#   run-tests.sh regress           tests/v1/engine and tests/v1/worker with the candidate
#                                  and on the untouched image (compare the failure lists)
#
# --confcutdir=tests/v1 skips the root tests/conftest.py (tblib is not in the image).
set -u
LAB=${LAB:-$HOME/spark-lab/late-admission}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
SUITES=(tests/v1/engine/test_late_admission.py tests/v1/worker/test_late_admission.py)
RELATED=(tests/v1/engine/test_structured_output_draft_handoff.py
         tests/v1/engine/test_compute_fairness_feedback.py
         tests/v1/worker/test_draft_tokens_handler.py
         tests/v1/worker/test_b12x_roce_health.py)

mounts() {
  local f
  for f in $(cd "$LAB/candidate" && find . -type f -name '*.py' | sed 's#^\./##'); do
    printf -- '-v %s/candidate/%s:%s/%s:ro ' "$LAB" "$f" "$V" "$f"
  done
}

run() {
  local mounted=$1 gpus=$2; shift 2
  local extra=()
  [ "$mounted" = candidate ] && extra=($(mounts))
  [ "$gpus" = gpu ] && extra+=(--gpus all)
  docker run --rm --network none --memory 48g -w "$V" "${extra[@]}" \
    --entrypoint python3 "$IMG" -m pytest -q -p no:cacheprovider --confcutdir=tests/v1 "$@"
}

case ${1:-} in
  sync)
    node=${2:-dgx4}
    here=$(cd "$(dirname "$0")/.." && pwd)
    tar -C "$here/overlay" -cf - . \
      | ssh "$node" "rm -rf ~/spark-lab/late-admission/candidate && mkdir -p ~/spark-lab/late-admission/candidate && tar -C ~/spark-lab/late-admission/candidate -xf -"
    ;;
  cpu)
    echo "== late admission (candidate, CPU)"
    run candidate cpu "${SUITES[@]}"
    ;;
  gpu)
    echo "== late admission and the batch-queue suites (candidate, GPU)"
    run candidate gpu "${SUITES[@]}" "${RELATED[@]}"
    ;;
  regress)
    for arm in candidate image; do
      echo "== tests/v1/engine and tests/v1/worker ($arm)"
      # Modules that need the network or ray fail to collect in both arms.
      run "$arm" gpu -rf --continue-on-collection-errors tests/v1/engine tests/v1/worker
    done
    ;;
  *) echo "usage: $0 sync [node]|cpu|gpu|regress" >&2; exit 2 ;;
esac
