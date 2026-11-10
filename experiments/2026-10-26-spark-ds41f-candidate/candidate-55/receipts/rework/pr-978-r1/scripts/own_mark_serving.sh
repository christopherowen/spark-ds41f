#!/usr/bin/env bash
# #978's mark on the serving path (exp/serving_path_bench.py), one session on sp8 under
# flock /tmp/spark-lab-gpu.lock: base (spark-ds41f:candidate-54 as built) and r1
# (candidate-55-r1's files mounted over it), order base r1 r1 base. --gpus all only for
# platform detection; no model is loaded.
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978
IMAGE=spark-ds41f:candidate-54
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
OUT=$WD/exp/receipts-serving; mkdir -p "$OUT"; log=$OUT/session.log
run() {
  local label=$1 arm=$2; local -a m=()
  if [[ -n $arm ]]; then
    while IFS= read -r f; do m+=(-v "$WD/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done \
      < <(cd "$WD/$arm" && find . -type f | sed 's|^\./||' | sort)
  fi
  echo "$(date -u +%FT%TZ) start $label arm=${arm:-none} mounts=$(( ${#m[@]} / 2 )) load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$log"
  local rc=0
  docker run --rm --gpus all --ipc=host -e HF_HUB_OFFLINE=1 -e DS41F_SNAPSHOT=$SNAP \
    -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$WD/exp:/opt/exp" "${m[@]}" \
    -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 "$IMAGE" \
    /opt/exp/serving_path_bench.py "$label" "/opt/exp/receipts-serving/$label.json" > "$OUT/$label.txt" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) end $label exit=$rc" | tee -a "$log"
}
[[ -e $HOME/spark-hold.json ]] && { echo "hold file present" | tee -a "$log"; exit 1; }
echo "$(date -u +%FT%TZ) session start, image $(docker image inspect $IMAGE --format '{{.Id}}')" | tee -a "$log"
run serving-base-boot1 ""
run serving-r1-boot1 arm-r1
run serving-r1-boot2 arm-r1
run serving-base-boot2 ""
echo "$(date -u +%FT%TZ) session end" | tee -a "$log"
