#!/usr/bin/env bash
# Usage: run_construct.sh <arm-dir|none> <label> [--gpu] [--image IMAGE]
# Runs exp/construct_async_llm.py in an image (default spark-ds41f:candidate-54) with
# the arm's files mounted over /opt/spark-ds41f/candidate/vllm. Output raw/<label>.{txt,json}
set -uo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978
ARM=$1; LABEL=$2; shift 2
GPU=(); IMAGE=spark-ds41f:candidate-54
while [[ $# -gt 0 ]]; do case $1 in --gpu) GPU=(--gpus all);; --image) IMAGE=$2; shift;; esac; shift; done
MOUNTS=()
if [[ $ARM != none ]]; then
  ARM=$(cd "$WD/$ARM" && pwd)
  while IFS= read -r f; do MOUNTS+=(-v "$ARM/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done \
    < <(cd "$ARM" && find . -type f | sed 's|^\./||' | sort)
fi
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
{ echo "# label=$LABEL image=$IMAGE ($(docker image inspect "$IMAGE" --format '{{.Id}}')) arm=$ARM gpu=${GPU[*]:-none} mounts=$(( ${#MOUNTS[@]} / 2 ))"
  echo "# started $(date -u +%FT%TZ)"; } > "$WD/raw/$LABEL.txt"
docker run --rm "${GPU[@]}" --ipc=host -e HF_HUB_OFFLINE=1 -e DS41F_SNAPSHOT=$SNAP \
  -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$WD/exp:/opt/exp" -v "$WD/raw:/opt/raw" "${MOUNTS[@]}" \
  -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 "$IMAGE" \
  /opt/exp/construct_async_llm.py "$LABEL" "/opt/raw/$LABEL.json" >> "$WD/raw/$LABEL.txt" 2>&1
code=$?
echo "# exit $code, finished $(date -u +%FT%TZ)" >> "$WD/raw/$LABEL.txt"
grep -E "^\[$LABEL\]|^# exit" "$WD/raw/$LABEL.txt"
