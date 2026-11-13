#!/bin/bash
# warmup_tokens.sh <arm> <boot>: the startup sampler warmup`s tokens with the arm`s six files mounted.
W=$HOME/spark-lab/agents/candidate-48-rework-980
IMG=sha256:d6ffd842978849651e756ebd39e9bf2da7ddb03e746113013533029878485a24
arm=$1; boot=$2
T=/opt/spark-ds41f/candidate/vllm
MNT=""
for f in sample/sampler.py sample/states.py spec_decode/acceptance_estimator.py spec_decode/adaptive_verification.py spec_decode/dflash/speculator.py spec_decode/speculator.py; do
  MNT="$MNT -v $W/ov/$arm/vllm/v1/worker/gpu/$f:$T/vllm/v1/worker/gpu/$f:ro"
done
exec flock /tmp/spark-lab-gpu.lock timeout 900 docker run --rm --name c48rw980-wt-$arm-$boot --gpus all --ipc=host --network=host \
  $MNT -v $W/warmup_tokens.py:/tmp/warmup_tokens.py:ro --entrypoint python3 $IMG /tmp/warmup_tokens.py --arm $arm --boot $boot
