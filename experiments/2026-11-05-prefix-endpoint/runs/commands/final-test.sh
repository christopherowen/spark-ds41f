#!/bin/bash
set -uo pipefail
cd "$HOME/spark-lab/agents/prefix-endpoint"
test ! -f runs/final-cpu.exit || exit 3
docker run --rm --name prefix-endpoint-cpu --runtime=runc --network=none --cpus=2 --memory=4g --memory-swap=4g -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e HF_HUB_OFFLINE=1 -e VLLM_PLUGINS= -e PYTHONPATH=/source -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v "$PWD/vllm:/source:ro" -v "$PWD/incoming:/probe:ro" -w /source --entrypoint python3 spark-ds41f:candidate-38 /probe/cpu_runner.py --noconftest -p no:cacheprovider tests/v1/core/test_prefix_caching.py tests/v1/core/test_single_type_kv_cache_manager.py -k 'not hisparse_async_admission_requires_only_import_destinations' -q > runs/final-cpu.txt 2>&1
result=$?
printf '%s\n' "$result" > runs/final-cpu.exit
exit "$result"
