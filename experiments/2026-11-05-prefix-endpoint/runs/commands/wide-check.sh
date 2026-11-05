#!/bin/bash
set -euo pipefail
cd "$HOME/spark-lab/agents/prefix-endpoint"
file=vllm/v1/core/kv_cache_coordinator.py
cp "vllm/$file" incoming/coordinator-fixed.py
restore() { cp incoming/coordinator-fixed.py "vllm/$file"; }
trap restore EXIT
git -C vllm show "HEAD:$file" > "vllm/$file"
bash test.sh before-prefix-suite 'not nonexistent_test' || true
restore
bash test.sh after-prefix-suite-focused 'not nonexistent_test' || true
