#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/c55-fusions/results /tmp/c55-fusions/caches

run_arm() {
  local label="$1"
  local img="$2"
  local wo="$3"
  local ffn="$4"
  local nxt="$5"
  local cache_name="$6"
  shift 6
  local extra_mounts=("$@")

  if [[ -f "/tmp/c55-fusions/results/${label}.json" ]]; then
    echo "=== [skip] ${label} already done ==="
    return 0
  fi

  local cache_dir="/tmp/c55-fusions/caches/${cache_name}"
  if [[ ! -d "$cache_dir" ]]; then
    mkdir -p "$cache_dir"
    if [[ "$cache_name" == c54* ]]; then
      cp -a /tmp/iket-diag/cache-c54/. "$cache_dir/"
    else
      cp -a /tmp/iket-diag/cache-c55/. "$cache_dir/"
    fi
  fi

  echo "=== [run] ${label} (WO=${wo} FFN=${ffn} NEXT=${nxt}) ==="
  flock -w 600 /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
    -v /tmp/iket-diag:/b:ro \
    -v /tmp/c55-fusions:/f \
    -v /home/chris/.cache/huggingface:/hf:ro \
    -v "${cache_dir}:/tcache" \
    "${extra_mounts[@]}" \
    -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm \
    -e TILELANG_CACHE_DIR=/tcache \
    -e VLLM_L2_PREFETCH_KERNELS=tilelang \
    -e VLLM_L2_PREFETCH_GRID=1 \
    -e VLLM_DS41_L2_PREFETCH_WO_MB="${wo}" \
    -e VLLM_DS41_L2_PREFETCH_FFN_MB="${ffn}" \
    -e VLLM_DS41_L2_PREFETCH_NEXT_MB="${nxt}" \
    --entrypoint python3 "${img}" /f/run_single_var_screen.py "${label}" "/f/results/${label}.json"
}

mount_dir_files() {
  local ovl_dir="$1"
  local container_base="$2"
  while IFS= read -r -d '' f; do
    local rel="${f#${ovl_dir}/}"
    printf -- "-v\n%s:%s/%s:ro\n" "$f" "$container_base" "$rel"
  done < <(find "$ovl_dir" -type f -print0)
}

# 1. Baselines & L2 budget candidates
run_arm "c54_l2cap"      "spark-ds41f:candidate-54" 15 12 12 "c54_base"
run_arm "c55_l2cap"      "spark-ds41f:candidate-55" 15 12 12 "c55_base"
run_arm "c55_l2_141517"  "spark-ds41f:candidate-55" 14 15 17 "c55_base"

# 2. Each fusion/kernel PR individually on top of c55_l2cap (15/12/12)
mapfile -t m994 < <(
  mount_dir_files /tmp/c55-fusions/overlays/pr994/tilelang /usr/local/lib/python3.12/dist-packages/tilelang
  mount_dir_files /tmp/c55-fusions/overlays/pr994/vllm /opt/spark-ds41f/candidate/vllm
)
run_arm "c55_pr994" "spark-ds41f:candidate-55" 15 12 12 "c55_pr994" "${m994[@]}"

mapfile -t m992 < <(mount_dir_files /tmp/c55-fusions/overlays/pr992 /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr992" "spark-ds41f:candidate-55" 15 12 12 "c55_pr992" "${m992[@]}"

mapfile -t m1078 < <(mount_dir_files /tmp/c55-fusions/overlays/pr1078 /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr1078" "spark-ds41f:candidate-55" 15 12 12 "c55_pr1078" "${m1078[@]}"

mapfile -t m1086 < <(mount_dir_files /tmp/c55-fusions/overlays/pr1086_k /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr1086k" "spark-ds41f:candidate-55" 15 12 12 "c55_pr1086k" "${m1086[@]}"

mapfile -t m1005 < <(mount_dir_files /tmp/c55-fusions/overlays/pr1005 /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr1005" "spark-ds41f:candidate-55" 15 12 12 "c55_pr1005" "${m1005[@]}"

mapfile -t m957p2 < <(mount_dir_files /tmp/c55-fusions/overlays/pr957p2 /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr957p2" "spark-ds41f:candidate-55" 15 12 12 "c55_pr957p2" "${m957p2[@]}"

mapfile -t m957p4 < <(mount_dir_files /tmp/c55-fusions/overlays/pr957p4 /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr957p4" "spark-ds41f:candidate-55" 15 12 12 "c55_pr957p4" "${m957p4[@]}"

mapfile -t m1081 < <(mount_dir_files /tmp/c55-fusions/overlays/pr1081 /opt/spark-ds41f/candidate/vllm)
run_arm "c55_pr1081" "spark-ds41f:candidate-55" 15 12 12 "c55_pr1081" "${m1081[@]}"

# 3. Repeat of c55_l2cap to bracket within-session drift
run_arm "c55_l2cap_r2" "spark-ds41f:candidate-55" 15 12 12 "c55_base"

echo "=== ALL 12 SINGLE-VARIABLE ARMS COMPLETE ==="
