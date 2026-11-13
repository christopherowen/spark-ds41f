#!/usr/bin/env bash
# The kernel cache key with the device compiler: candidate-55 on a cache candidate-54 filled (CUDA 13.0.2 kernels),
# first with the image as built (it loads them), then with the patched key (it must compile its own).
set -euo pipefail
A=$HOME/spark-lab/agents/candidate-55-m24
T=/usr/local/lib/python3.12/dist-packages/tilelang
count() { echo "$1: kernels $(ls "$2"/*snapshot*/linux-aarch64/kernels 2>/dev/null | wc -l) cuda-binaries $(ls "$2"/*snapshot*/cuda-binaries 2>/dev/null | wc -l)"; }
run_arm() {
  local label="$1" cache="$2"; shift 2
  echo "=== $label"; count "before" "$cache"
  local t0=$SECONDS
  flock -w 3600 /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
    -v /tmp/iket-diag:/b:ro -v "$A":/f -v /home/chris/.cache/huggingface:/hf:ro -v "${cache}:/tcache" \
    -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm -e TILELANG_CACHE_DIR=/tcache \
    -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_L2_PREFETCH_GRID=1 \
    -e VLLM_DS41_L2_PREFETCH_WO_MB=15 -e VLLM_DS41_L2_PREFETCH_FFN_MB=12 -e VLLM_DS41_L2_PREFETCH_NEXT_MB=12 \
    "$@" --entrypoint python3 spark-ds41f:candidate-55 /f/run_single_var_screen.py "${label}" "/f/results/${label}.json" > "$A/log-${label}.txt" 2>&1 || echo "FAILED $label"
  echo "wall $((SECONDS - t0)) s"; count "after" "$cache"
  grep -E "Prepared [0-9]+ TileLang calls" "$A/log-${label}.txt" | sed "s/^.*\] //" | cut -c1-90 || true
  grep -E "^\s*\[${label}" "$A/log-${label}.txt" | grep -E "1x2@|8x16|8x24|16x32|16x48" || tail -5 "$A/log-${label}.txt"
}
rm -rf "$A/caches/keytest"; cp -a "$A/caches/c54_base" "$A/caches/keytest"
run_arm key_asbuilt_on_c54_cache "$A/caches/keytest"
run_arm key_patched_on_c54_cache "$A/caches/keytest" -v "$A/kernel_cache.py:$T/cache/kernel_cache.py:ro"
run_arm key_patched_again        "$A/caches/keytest" -v "$A/kernel_cache.py:$T/cache/kernel_cache.py:ro"
echo ALL DONE 6
