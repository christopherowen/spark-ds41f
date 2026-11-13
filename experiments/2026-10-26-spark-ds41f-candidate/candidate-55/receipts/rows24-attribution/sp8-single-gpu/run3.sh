#!/usr/bin/env bash
# Third pass: both builds with the DS4.1 L2 prefetch off.
set -euo pipefail
A=$HOME/spark-lab/agents/candidate-55-m24
run_arm() {
  local label="$1" img="$2" cache="$3"; shift 3
  [[ -f "$A/results/${label}.json" ]] && { echo "skip $label"; return 0; }
  echo "=== $label"
  flock -w 1800 /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
    -v /tmp/iket-diag:/b:ro -v "$A":/f -v /home/chris/.cache/huggingface:/hf:ro -v "$A/caches/${cache}:/tcache" \
    -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm -e TILELANG_CACHE_DIR=/tcache \
    -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_L2_PREFETCH_GRID=1 \
    -e VLLM_DS41_L2_PREFETCH_WO_MB=15 -e VLLM_DS41_L2_PREFETCH_FFN_MB=12 -e VLLM_DS41_L2_PREFETCH_NEXT_MB=12 \
    "$@" --entrypoint python3 "${img}" /f/run_single_var_screen.py "${label}" "/f/results/${label}.json" 2>&1 | grep --line-buffered -E "^\s*\[|Error|Traceback" || true
}
run_arm c54_l2off_1 spark-ds41f:candidate-54 c54_base -e VLLM_DS41_L2_PREFETCH=0
run_arm c55_l2off_1 spark-ds41f:candidate-55 c55_base -e VLLM_DS41_L2_PREFETCH=0
run_arm c55_l2off_2 spark-ds41f:candidate-55 c55_base -e VLLM_DS41_L2_PREFETCH=0
run_arm c54_l2off_2 spark-ds41f:candidate-54 c54_base -e VLLM_DS41_L2_PREFETCH=0
echo ALL DONE 3
