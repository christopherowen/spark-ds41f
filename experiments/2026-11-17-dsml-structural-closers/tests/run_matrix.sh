#!/bin/bash
# run_matrix.sh <IMAGE> [OVERLAY_DIR]
#   IMAGE        serving image tag, e.g. spark-ds41f:candidate-60
#   OVERLAY_DIR  directory holding vllm/parser/deepseek_v41.py (default: ../overlay)
#
# Offline DSML tool-parser matrix in the candidate-60 image: CPU only, --network none,
# no GPU, no server. Runs stock, patched and kill-switch arms, then gates them with
# compare.py (G1-G6 plus the instrument check).
#
# The candidate-60 tree is expected at /opt/spark-ds41f/candidate/vllm (PYTHONPATH), so the
# parser file the overlay replaces is /opt/spark-ds41f/candidate/vllm/vllm/parser/deepseek_v41.py.
# MODEL_DIR is mounted read-only at /model and only its tokenizer files are read.
#
#   MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017   (override with MODEL_DIR=...)
#   OUT=./out                                              (override with OUT=...)
set -euo pipefail
IMAGE=${1:?usage: run_matrix.sh <IMAGE> [OVERLAY_DIR]}
HERE=$(cd "$(dirname "$0")" && pwd)
OVERLAY_DIR=${2:-$HERE/../overlay}
OVERLAY=$OVERLAY_DIR/vllm/parser/deepseek_v41.py
[ -f "$OVERLAY" ] || { echo "no overlay file at $OVERLAY" >&2; exit 2; }
MODEL_DIR=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017}
[ -d "$MODEL_DIR" ] || { echo "no checkpoint dir at $MODEL_DIR" >&2; exit 2; }
TREE=${TREE:-/opt/spark-ds41f/candidate/vllm}
TARGET=$TREE/vllm/parser/deepseek_v41.py
K=VLLM_DSV41_STRUCTURAL_DSML
OUT=${OUT:-$PWD/out}
mkdir -p "$OUT"

echo "image=$IMAGE overlay=$OVERLAY model=$MODEL_DIR tree=$TREE" | tee "$OUT/run.txt"
docker image inspect -f 'labels: vllm.tree={{index .Config.Labels "local.spark-ds41f.vllm.tree"}}' "$IMAGE" 2>/dev/null | tee -a "$OUT/run.txt" || true

base=(docker run --rm --network none --entrypoint python3
      -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH="$TREE" -e MODEL_DIR=/model
      -v "$MODEL_DIR:/model:ro" -v "$HERE:/p:ro")
ov=(-v "$OVERLAY:$TARGET:ro")

"${base[@]}" "$IMAGE" /p/matrix.py > "$OUT/stock.jsonl" 2> "$OUT/stock.err"
"${base[@]}" "${ov[@]}" -e $K=1 "$IMAGE" /p/matrix.py > "$OUT/patched.jsonl" 2> "$OUT/patched.err"
"${base[@]}" "${ov[@]}" -e $K=0 "$IMAGE" /p/matrix.py > "$OUT/killswitch.jsonl" 2> "$OUT/killswitch.err"

echo "--- preflight ---" | tee -a "$OUT/run.txt"
for arm in stock patched killswitch; do
  printf '%-11s %s\n' "$arm" "$(grep -m1 '"preflight"' "$OUT/$arm.err" || echo '(none)')" | tee -a "$OUT/run.txt"
done
python3 "$HERE/compare.py" "$OUT/stock.jsonl" "$OUT/patched.jsonl" "$OUT/killswitch.jsonl" | tee "$OUT/VERDICT.txt"
