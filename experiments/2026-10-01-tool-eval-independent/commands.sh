#!/bin/sh
# Record of authorized benchmark commands. Do not rerun without an exclusive window.
set -eu
BENCH=/tmp/spark3-tool-eval-20261001/.venv/bin/tool-eval-bench

"$BENCH" run --model deepseek-v4.1-flash --backend vllm \
  --base-url http://10.0.1.71:8000/v1 --hardmode --temperature 0 --seed 42 \
  --parallel 1 --trials 1 --max-turns 8 --timeout 120 --reference-date 2026-10-01 \
  --backend-kwargs '{"max_tokens":16384,"chat_template_kwargs":{"thinking":true}}' \
  --no-warmup --no-preflight --no-probe-engine --no-live \
  --label r5o-independent-c1-greedy --output-dir runs/full \
  --json-file runs/full-results.json > runs/full-client.log 2>&1

"$BENCH" run --model deepseek-v4.1-flash --backend vllm \
  --base-url http://10.0.1.71:8000/v1 --hardmode --temperature 0 --seed 42 \
  --parallel 1 --trials 2 --max-turns 8 --timeout 120 --reference-date 2026-10-01 \
  --backend-kwargs '{"max_tokens":16384,"chat_template_kwargs":{"thinking":true}}' \
  --no-warmup --no-preflight --no-probe-engine --no-live \
  --scenarios TC-23 TC-38 TC-43 TC-50 TC-57 TC-58 TC-67 TC-68 TC-74 TC-81 TC-85 TC-88 \
  --label r5o-independent-failure-repeats --output-dir runs/repeats \
  --json-file runs/repeat-results.json > runs/repeat-client.log 2>&1

"$BENCH" run --model deepseek-v4.1-flash --backend vllm \
  --base-url http://10.0.1.71:8000/v1 --hardmode --temperature 0 --seed 42 \
  --parallel 1 --trials 3 --max-turns 8 --timeout 120 --reference-date 2026-10-01 \
  --backend-kwargs '{"max_tokens":16384,"chat_template_kwargs":{"thinking":true,"drop_thinking":false}}' \
  --no-warmup --no-preflight --no-probe-engine --no-live --scenarios TC-88 \
  --label r5o-tc88-preserve-reasoning --output-dir runs/preserve-reasoning \
  --json-file runs/preserve-reasoning-results.json > runs/preserve-reasoning-client.log 2>&1
