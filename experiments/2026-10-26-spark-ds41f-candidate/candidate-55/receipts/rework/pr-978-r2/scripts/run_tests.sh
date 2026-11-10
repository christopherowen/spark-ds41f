#!/usr/bin/env bash
# Usage: run_tests.sh <arm-dir> <label> [--gpu] -- <pytest args...>
# Runs pytest in spark-ds41f:candidate-54 with every file under <arm-dir>
# mounted read-only over the image's vLLM tree (/opt/spark-ds41f/candidate/vllm),
# with the release-test bundle's plugins and environment. Output: raw/<label>.txt
set -euo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r2
ARM=$(cd "$1" && pwd); LABEL=$2; shift 2
GPU=()
if [[ "${1:-}" == "--gpu" ]]; then GPU=(--gpus all); shift; fi
[[ "${1:-}" == "--" ]] && shift
IMAGE=${IMAGE:-spark-ds41f:candidate-54}
MOUNTS=()
while IFS= read -r f; do
  MOUNTS+=(-v "$ARM/$f:/opt/spark-ds41f/candidate/vllm/$f:ro")
done < <(cd "$ARM" && find . -type f | sed 's|^\./||' | sort)
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
mkdir -p "$WD/raw" "$WD/cache"
{
  echo "# label=$LABEL image=$IMAGE ($(docker image inspect "$IMAGE" --format '{{.Id}}')) arm=$ARM gpu=${GPU[*]:-none}"
  echo "# mounted: $(cd "$ARM" && find . -type f | wc -l) files; sha256 of each:"
  (cd "$ARM" && find . -type f | sort | xargs sha256sum | sed 's/^/#   /')
  echo "# started $(date -u +%FT%TZ)"
} > "$WD/raw/$LABEL.txt"
set +e
docker run --rm "${GPU[@]}" --ipc=host --shm-size=8g \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x \
  -e DS41F_SNAPSHOT=$SNAP -e TILELANG_CACHE_DIR=/c/tilelang \
  -v "$WD/bundle:/b:ro" -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$WD/cache:/c" \
  "${MOUNTS[@]}" -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 "$IMAGE" \
  -m pytest -q -rfEs -p no:cacheprovider -p tblib_stub -p offline_skip -p deepseek_models \
  -o faulthandler_timeout=300 --basetemp=/c/pytest-tmp-$LABEL "$@" >> "$WD/raw/$LABEL.txt" 2>&1
code=$?
set -e
echo "# exit $code, finished $(date -u +%FT%TZ)" >> "$WD/raw/$LABEL.txt"
tail -n 40 "$WD/raw/$LABEL.txt"
exit 0
