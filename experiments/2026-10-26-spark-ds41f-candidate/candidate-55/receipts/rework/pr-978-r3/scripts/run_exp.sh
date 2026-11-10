#!/usr/bin/env bash
# Usage: run_exp.sh <arm|none> <log> <script> [args...]
# One run of exp/<script> in spark-ds41f:candidate-54 with arms/<arm>/* mounted read-only
# over /opt/spark-ds41f/candidate/vllm ("none" = candidate-54 as built; arm paths may be
# absolute). --gpus all. Takes /tmp/spark-lab-gpu.lock itself after checking that
# ~/spark-hold.json is absent; do not wrap it in another flock.
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r3
IMAGE=${IMAGE:-spark-ds41f:candidate-54}
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
ARM=$1; ARMNAME=$1; LOG=$2; shift 2
m=()
if [[ $ARM != none ]]; then
  [[ $ARM = /* ]] || ARM=$WD/arms/$ARM
  while IFS= read -r f; do
    m+=(-v "$ARM/$f:/opt/spark-ds41f/candidate/vllm/$f:ro")
  done < <(cd "$ARM" && find . -type f | sed 's|^\./||' | sort)
fi
mkdir -p "$(dirname "$LOG")"
exec 9>/tmp/spark-lab-gpu.lock
flock 9
if [[ -e $HOME/spark-hold.json ]]; then echo "hold file present; not run" | tee -a "$LOG"; exit 1; fi
{
  echo "# $(date -u +%FT%TZ) arm=$ARMNAME mounts=$(( ${#m[@]} / 2 )) image=$IMAGE ($(docker image inspect "$IMAGE" --format '{{.Id}}')) load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "# exp/$*"
} >> "$LOG"
docker run --rm --gpus all --ipc=host -e HF_HUB_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e DS41F_SNAPSHOT=$SNAP -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/b ${EXTRA_ENV:-} \
  -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$WD/exp:/opt/exp" -v "$WD/cache:/c" \
  -v "$HOME/spark-lab/agents/candidate-55-rework-978-r2/bundle:/b:ro" \
  "${m[@]}" -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 "$IMAGE" "/opt/exp/$@" >> "$LOG" 2>&1
rc=$?
echo "# $(date -u +%FT%TZ) exit $rc" >> "$LOG"
exit $rc
