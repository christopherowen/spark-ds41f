#!/bin/bash
# The patch's test file in a GPU container (through run.sh: lock, hold-file and docker-ps checks).
# usage: pytest.sh <name> <arm: c54|c57p5>
set -euo pipefail
W=$HOME/spark-lab/agents/round3-957p5
name=$1; arm=$2
V=()
while read -r f; do V+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < "$W/in/changed-$arm.txt"
exec "$W/scripts/run.sh" "$name" --rm --name "r3-957p5-$name" --gpus all --network none --ipc=host \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  "${V[@]}" -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 spark-ds41f:candidate-50 \
  -m pytest --noconftest -p no:cacheprovider -v tests/v1/spec_decode/test_adaptive_verification.py
