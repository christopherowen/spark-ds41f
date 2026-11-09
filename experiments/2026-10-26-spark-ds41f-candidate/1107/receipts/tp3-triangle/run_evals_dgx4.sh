#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""
cd /home/swank/projects/spark-ds41f
EVAL_DIR=/home/swank/spark-lab/1107-tp3-triangle/evals
mkdir -p "$EVAL_DIR/private/longbench-v2" "$EVAL_DIR/private/ruler" "$EVAL_DIR/private/humanevalplus" "$EVAL_DIR/vision-eval" "$EVAL_DIR/tool-eval-bench"

echo "=== [EVAL 1/5] LongBench v2 ==="
rm -rf "$EVAL_DIR/private/longbench-v2"/* "$EVAL_DIR/longbench-v2.json"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-11-05-answer-quality/evaluate.py \
  --url http://10.0.1.71:8000 \
  --cache /home/swank/spark-lab/datasets/longbench-v2/ac35867b9c9436dc81c4c7535fe640ba7cf47877/policy-65536-v1 \
  --split pilot \
  --private "$EVAL_DIR/private/longbench-v2" \
  --out "$EVAL_DIR/longbench-v2.json" || echo "LongBench v2 exit $?"

echo "=== [EVAL 2/5] RULER ==="
rm -rf "$EVAL_DIR/private/ruler"/* "$EVAL_DIR/ruler.json"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-11-05-context-quality/evaluate.py \
  --url http://10.0.1.71:8000 \
  --cache /home/swank/spark-lab/datasets/ruler/db43f92ff634bbce1c33d1fb15858aec4bba509a/policy-65536-v1/pilot \
  --split pilot \
  --stage screen \
  --private "$EVAL_DIR/private/ruler" \
  --out "$EVAL_DIR/ruler.json" || echo "RULER exit $?"

echo "=== [EVAL 3/5] EvalPlus / HumanEvalPlus ==="
rm -rf "$EVAL_DIR/private/humanevalplus"/* "$EVAL_DIR/humanevalplus-generation.json" "$EVAL_DIR/humanevalplus-grade.json"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-11-05-answer-quality/generate_code.py \
  --url http://10.0.1.71:8000 \
  --cache /home/swank/spark-lab/datasets/evalplus/v0.1.10 \
  --concurrency 8 \
  --private "$EVAL_DIR/private/humanevalplus" \
  --out "$EVAL_DIR/humanevalplus-generation.json" || echo "EvalPlus gen exit $?"
cp -f "$EVAL_DIR/private/humanevalplus/samples.jsonl" "$EVAL_DIR/humanevalplus-samples.jsonl" || true
docker run --rm --user "$(id -u):$(id -g)" --runtime=runc --network none --read-only \
  --cpus 2 --memory 4g --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
  --env SPARK_EVALPLUS_ISOLATED=1 --env OPENBLAS_NUM_THREADS=1 \
  --tmpfs /tmp:rw,nosuid,size=128m \
  --mount type=bind,source=/home/swank/projects/spark-ds41f/experiments/2026-11-05-answer-quality,target=/work,readonly \
  --mount type=bind,source=/home/swank/spark-lab/datasets/evalplus/v0.1.10,target=/data,readonly \
  --mount type=bind,source="$EVAL_DIR/private/humanevalplus",target=/samples,readonly \
  --mount type=bind,source="$EVAL_DIR",target=/results \
  --entrypoint python3 spark-ds41f:1107 \
  /work/code_quality.py grade --cache /data --samples /samples/samples.jsonl --out /results/humanevalplus-grade.json || echo "EvalPlus grade exit $?"

echo "=== [EVAL 4/5] Vision Eval (2,283 items) ==="
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-10-25-vision-eval/vision_eval.py \
  --concurrency 8 http://10.0.1.71:8000 /home/swank/spark-lab/vision-eval/data \
  > "$EVAL_DIR/vision-eval/1107-tp3-triangle.jsonl" || echo "Vision Eval exit $?"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-10-25-vision-eval/score.py \
  /home/swank/spark-lab/vision-eval/data "$EVAL_DIR/vision-eval/1107-tp3-triangle.jsonl" \
  > "$EVAL_DIR/vision-eval/score.json" || echo "Vision score exit $?"

echo "=== [EVAL 5/5] tool-eval-bench ==="
rm -rf "$EVAL_DIR/tool-eval-bench"/* "$EVAL_DIR/tool-eval-bench.json"
CUDA_VISIBLE_DEVICES="" /home/swank/spark-lab/tool-eval-bench-latest/.venv/bin/tool-eval-bench \
  --base-url http://10.0.1.71:8000/v1 \
  --model deepseek-v4.1-flash \
  --hardmode --temperature 0 --seed 42 --parallel 8 \
  --json-file "$EVAL_DIR/tool-eval-bench.json" \
  --output-dir "$EVAL_DIR/tool-eval-bench" || echo "tool-eval-bench exit $?"

echo "=== ALL 5 EVALS COMPLETE ==="

echo "=== [EVAL 1/5 RETRY] LongBench v2 ==="
rm -rf "$EVAL_DIR/private/longbench-v2" "$EVAL_DIR/longbench-v2.json"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-11-05-answer-quality/evaluate.py \
  --url http://10.0.1.71:8000 \
  --cache /home/swank/spark-lab/datasets/longbench-v2/ac35867b9c9436dc81c4c7535fe640ba7cf47877/policy-65536-v1 \
  --split pilot \
  --private "$EVAL_DIR/private/longbench-v2" \
  --out "$EVAL_DIR/longbench-v2.json" || echo "LongBench v2 exit $?"

echo "=== [EVAL 2/5 RETRY] RULER ==="
rm -rf "$EVAL_DIR/private/ruler" "$EVAL_DIR/ruler.json"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-11-05-context-quality/evaluate.py \
  --url http://10.0.1.71:8000 \
  --cache /home/swank/spark-lab/datasets/ruler/db43f92ff634bbce1c33d1fb15858aec4bba509a/policy-65536-v1/pilot \
  --split pilot \
  --stage screen \
  --private "$EVAL_DIR/private/ruler" \
  --out "$EVAL_DIR/ruler.json" || echo "RULER exit $?"

echo "=== [EVAL 3/5 RETRY] EvalPlus / HumanEvalPlus ==="
rm -rf "$EVAL_DIR/private/humanevalplus" "$EVAL_DIR/humanevalplus-generation.json" "$EVAL_DIR/humanevalplus-grade.json"
CUDA_VISIBLE_DEVICES="" python3 experiments/2026-11-05-answer-quality/generate_code.py \
  --url http://10.0.1.71:8000 \
  --cache /home/swank/spark-lab/datasets/evalplus/v0.1.10 \
  --concurrency 8 \
  --private "$EVAL_DIR/private/humanevalplus" \
  --out "$EVAL_DIR/humanevalplus-generation.json" || echo "EvalPlus gen exit $?"
cp -f "$EVAL_DIR/private/humanevalplus/samples.jsonl" "$EVAL_DIR/humanevalplus-samples.jsonl" || true
docker run --rm --user "$(id -u):$(id -g)" --runtime=runc --network none --read-only \
  --cpus 2 --memory 4g --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
  --env SPARK_EVALPLUS_ISOLATED=1 --env OPENBLAS_NUM_THREADS=1 \
  --tmpfs /tmp:rw,nosuid,size=128m \
  --mount type=bind,source=/home/swank/projects/spark-ds41f/experiments/2026-11-05-answer-quality,target=/work,readonly \
  --mount type=bind,source=/home/swank/spark-lab/datasets/evalplus/v0.1.10,target=/data,readonly \
  --mount type=bind,source="$EVAL_DIR/private/humanevalplus",target=/samples,readonly \
  --mount type=bind,source="$EVAL_DIR",target=/results \
  --entrypoint python3 spark-ds41f:1107 \
  /work/code_quality.py grade --cache /data --samples /samples/samples.jsonl --out /results/humanevalplus-grade.json || echo "EvalPlus grade exit $?"

echo "=== ALL 5 EVALS RETRY COMPLETE ==="
