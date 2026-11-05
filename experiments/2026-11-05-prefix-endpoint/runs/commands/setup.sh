#!/bin/bash
set -euxo pipefail
cd "$HOME/spark-lab/agents/prefix-endpoint"
test ! -d vllm
source_repo=$(cat source-object-repository.txt)
git -c pack.threads=1 clone --shared --no-checkout "$source_repo" vllm
git -C vllm remote set-url origin https://github.com/christopherowen/vllm.git
git -C vllm remote add upstream https://github.com/local-inference-lab/vllm.git
git -C vllm checkout -b prefix-endpoint 1f33102547e4da5c64712390604695d1abd577c4
git -C vllm apply ../incoming/regression.patch
docker image inspect spark-ds41f:candidate-38 --format '{{.Id}} {{index .Config.Labels "local.spark-ds41f.vllm.tree"}}' > runs/image.txt
set +e
docker run --rm --name prefix-endpoint-cpu --runtime=runc --network=none --cpus=2 --memory=4g --memory-swap=4g -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e HF_HUB_OFFLINE=1 -e VLLM_PLUGINS= -e PYTHONPATH=/source -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v "$PWD/vllm:/source:ro" -w /source --entrypoint python spark-ds41f:candidate-38 -m pytest --noconftest -p no:cacheprovider tests/v1/core/test_prefix_caching.py -k ced_retains_encoder_endpoint -q > runs/before.txt 2>&1
result=$?
printf '%s\n' "$result" > runs/before.exit
exit "$result"
