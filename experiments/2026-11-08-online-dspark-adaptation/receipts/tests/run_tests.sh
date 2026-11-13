#!/bin/bash
# Run the online-adaptation tests (and the affected spec-decode suites) in the
# candidate-54 image with the changed files of the prepared tree mounted over it.
set -u
TREE=$HOME/spark-lab/agents/online-dspark/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-55b552a4/src/vllm
OUT=$HOME/spark-lab/agents/online-dspark/out
mkdir -p $OUT
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/*)
ROOT=/opt/spark-ds41f/candidate/vllm
MOUNTS=""
for f in vllm/v1/worker/gpu/spec_decode/online_dspark_adaptation.py vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py \
         vllm/v1/worker/gpu/spec_decode/dspark/speculator.py vllm/v1/worker/gpu/spec_decode/speculator.py \
         vllm/v1/worker/gpu/spec_decode/copy_drafts.py vllm/v1/worker/gpu/model_runner.py \
         tests/v1/spec_decode/test_online_dspark_adaptation.py; do
  MOUNTS="$MOUNTS -v $TREE/$f:$ROOT/$f:ro"
done
TESTS=${TESTS:-tests/v1/spec_decode/test_online_dspark_adaptation.py}
LOG=${LOG:-$OUT/pytest.txt}
flock /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host --shm-size=8g $MOUNTS \
  -v $HOME/.cache/huggingface/hub:$HOME/.cache/huggingface/hub:ro -e DS41F_SNAPSHOT=$SNAP -e HF_HUB_OFFLINE=1 -e COLUMNS=120 \
  -v $HOME/.cache/spark-lab/tilelang-tests:/root/.tilelang \
  -w $ROOT --entrypoint python3 spark-ds41f:candidate-54 \
  -m pytest --noconftest -q -p no:cacheprovider $TESTS ${PYTEST_ARGS:-} > $LOG 2>&1
echo "exit $?" >> $LOG
tail -30 $LOG
