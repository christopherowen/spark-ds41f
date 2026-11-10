#!/usr/bin/env bash
# Usage: serving_session.sh <outdir-name> <label>:<arm>[:<ENV=V,...>] ...
# One hold of flock /tmp/spark-lab-gpu.lock (taken here, after checking that
# ~/spark-hold.json is absent), running exp/$BENCH (default: round 1's real-AsyncLLM
# frontend bench, serving_path_bench.py) once per argument, in the order given, in
# spark-ds41f:candidate-54 with arms/<arm>/* mounted read-only over
# /opt/spark-ds41f/candidate/vllm ("none" = candidate-54 as built).
# --gpus all is for platform detection only; no model is loaded.
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r3
IMAGE=spark-ds41f:candidate-54
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
OUTN=$1; shift
OUT=$WD/raw/$OUTN; mkdir -p "$OUT" "$WD/exp/out/$OUTN"; log=$OUT/session.log
BENCH=${BENCH:-serving_path_bench.py}

run() {
  local spec=$1 label arm envs
  IFS=: read -r label arm envs <<< "$spec"
  local -a m=() e=()
  if [[ $arm != none ]]; then
    while IFS= read -r f; do
      m+=(-v "$WD/arms/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro")
    done < <(cd "$WD/arms/$arm" && find . -type f | sed 's|^\./||' | sort)
  fi
  if [[ -n ${envs:-} ]]; then
    IFS=, read -ra kv <<< "$envs"
    for x in "${kv[@]}"; do e+=(-e "$x"); done
  fi
  echo "$(date -u +%FT%TZ) start $label arm=$arm mounts=$(( ${#m[@]} / 2 )) env=${envs:-} load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$log"
  local rc=0
  docker run --rm --gpus all --ipc=host -e HF_HUB_OFFLINE=1 -e DS41F_SNAPSHOT=$SNAP "${e[@]}" \
    -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$WD/exp:/opt/exp" "${m[@]}" \
    -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 "$IMAGE" \
    "/opt/exp/$BENCH" "$label" "/opt/exp/out/$OUTN/$label.json" > "$OUT/$label.txt" 2>&1 || rc=$?
  cp "$WD/exp/out/$OUTN/$label.json" "$OUT/" 2>/dev/null
  echo "$(date -u +%FT%TZ) end $label exit=$rc" | tee -a "$log"
}

exec 9>/tmp/spark-lab-gpu.lock
echo "$(date -u +%FT%TZ) waiting for the GPU lock" | tee -a "$log"
flock 9
if [[ -e $HOME/spark-hold.json ]]; then
  echo "$(date -u +%FT%TZ) hold file present; not running" | tee -a "$log"
  exit 1
fi
echo "$(date -u +%FT%TZ) lock taken; session start, image $(docker image inspect $IMAGE --format '{{.Id}}')" | tee -a "$log"
for spec in "$@"; do run "$spec"; done
echo "$(date -u +%FT%TZ) session end" | tee -a "$log"
