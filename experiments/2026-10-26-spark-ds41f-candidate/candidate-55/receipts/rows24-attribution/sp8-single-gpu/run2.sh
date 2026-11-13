#!/usr/bin/env bash
# Second pass: vLLM 0801 core partition off, and the pre-0800 loader files (smoke of the ring overlay).
set -euo pipefail
A=$HOME/spark-lab/agents/candidate-55-m24
while pgrep -f "$A/run.sh" >/dev/null; do sleep 10; done
run_arm() {
  local label="$1" img="$2" cache="$3"; shift 3
  [[ -f "$A/results/${label}.json" ]] && { echo "skip $label"; return 0; }
  local cdir="$A/caches/${cache}"
  echo "=== $label"
  flock -w 1800 /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
    -v /tmp/iket-diag:/b:ro -v "$A":/f -v /home/chris/.cache/huggingface:/hf:ro -v "${cdir}:/tcache" \
    -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm -e TILELANG_CACHE_DIR=/tcache \
    -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_L2_PREFETCH_GRID=1 \
    -e VLLM_DS41_L2_PREFETCH_WO_MB=15 -e VLLM_DS41_L2_PREFETCH_FFN_MB=12 -e VLLM_DS41_L2_PREFETCH_NEXT_MB=12 \
    "$@" --entrypoint python3 "${img}" /f/run_single_var_screen.py "${label}" "/f/results/${label}.json" 2>&1 | grep --line-buffered -E "^\s*\[|LLM ready|Error|Traceback|affinity" || true
}
L=(); for f in base_loader direct_loader dummy_loader weight_utils; do L+=(-v "$A/ovl/no1134-loader/overlay/vllm/model_executor/model_loader/$f.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/model_loader/$f.py:ro"); done
run_arm c55_noaffinity_1 spark-ds41f:candidate-55 c55_base -e VLLM_ROCE_CPU_AFFINITY=0
run_arm c55_c54loader_1  spark-ds41f:candidate-55 c55_base "${L[@]}"
run_arm c55_3            spark-ds41f:candidate-55 c55_base
run_arm c55_noaffinity_2 spark-ds41f:candidate-55 c55_base -e VLLM_ROCE_CPU_AFFINITY=0
run_arm c55_c54loader_2  spark-ds41f:candidate-55 c55_base "${L[@]}"
echo ALL DONE 2
