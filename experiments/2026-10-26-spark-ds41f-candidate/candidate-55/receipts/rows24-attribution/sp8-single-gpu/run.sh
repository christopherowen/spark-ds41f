#!/usr/bin/env bash
# Single-GPU screen of the weight arena (vLLM 0800) at a TP4 rank shapes, dummy weights, 15/12/12 MB caps.
set -euo pipefail
A=$HOME/spark-lab/agents/candidate-55-m24
run_arm() {
  local label="$1" img="$2" cache="$3" arena="$4"
  [[ -f "$A/results/${label}.json" ]] && { echo "skip $label"; return 0; }
  local cdir="$A/caches/${cache}"
  [[ -d "$cdir" ]] || { mkdir -p "$cdir"; cp -a "/tmp/c55-fusions/caches/${cache}/." "$cdir/"; }
  echo "=== $label"
  flock -w 1800 /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
    -v /tmp/iket-diag:/b:ro -v "$A":/f -v /home/chris/.cache/huggingface:/hf:ro -v "${cdir}:/tcache" \
    -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm -e TILELANG_CACHE_DIR=/tcache \
    -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_L2_PREFETCH_GRID=1 \
    -e VLLM_DS41_L2_PREFETCH_WO_MB=15 -e VLLM_DS41_L2_PREFETCH_FFN_MB=12 -e VLLM_DS41_L2_PREFETCH_NEXT_MB=12 \
    -e VLLM_DUMMY_WEIGHT_ARENA="${arena}" \
    --entrypoint python3 "${img}" /f/run_single_var_screen.py "${label}" "/f/results/${label}.json" 2>&1 | grep -E "^\s*\[|LLM ready|Error|Traceback|arena" || true
}
run_arm c54_1          spark-ds41f:candidate-54 c54_base 1
run_arm c55_1          spark-ds41f:candidate-55 c55_base 1
run_arm c55_noarena_1  spark-ds41f:candidate-55 c55_base 0
run_arm c55_noarena_2  spark-ds41f:candidate-55 c55_base 0
run_arm c55_2          spark-ds41f:candidate-55 c55_base 1
run_arm c54_2          spark-ds41f:candidate-54 c54_base 1
echo ALL DONE
