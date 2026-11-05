#!/bin/bash
set -uo pipefail
cd "$HOME/spark-lab/agents/prefix-endpoint"
arm=$1
filter=${2:-ced_retains_encoder_endpoint}
test -f runs/fresh-series.json || exit 2
test ! -f "runs/$arm.exit" || exit 3
/usr/bin/time -v docker run --rm --name prefix-endpoint-cpu --runtime=runc --network=none --cpus=2 --memory=4g --memory-swap=4g -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e HF_HUB_OFFLINE=1 -e VLLM_PLUGINS= -e PYTHONPATH=/source -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v "$PWD/vllm:/source:ro" -w /source --entrypoint python3 spark-ds41f:candidate-38 -m pytest --noconftest -p no:cacheprovider tests/v1/core/test_prefix_caching.py -k "$filter" -q > "runs/$arm.txt" 2> "runs/$arm.stderr"
result=$?
printf '%s\n' "$result" > "runs/$arm.exit"
exit "$result"
