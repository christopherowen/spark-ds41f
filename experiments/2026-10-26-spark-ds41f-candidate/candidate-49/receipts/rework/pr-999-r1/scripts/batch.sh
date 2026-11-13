#!/usr/bin/env bash
# PR #999 rework r1 batch: served-configuration enumeration, then the gap bench on two boots.
set -u
B=$HOME/spark-lab/agents/candidate-49-rework-999
C=sha256:9f51f2e2c5ab728e96490d794bff7b83a147d6d3d61333c5b60b2cff662049cd
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
HFC=$(readlink -f /home/chris/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)
J=$B/out/json; mkdir -p $J
# 1. The IndexerTopK configurations the image declares for every indexed layer and shape (serving env: VLLM_DS41_LSA unset)
$B/run.sh served-topk-configs $C -- flock $L docker run --rm --gpus all --ipc=host --name c49rw999-enum --entrypoint "" -w $V -v $B/src/served_topk_configs.py:/tmp/served_topk_configs.py:ro -v $HFC:/tmp/hf_config.json:ro -v $J:/out $C bash -lc "python3 /tmp/served_topk_configs.py /tmp/hf_config.json /out/served-topk-configs.json"
# 1b. Same with the unserved LSA development flag, to show the cell is absent there too
$B/run.sh served-topk-configs-lsa1 $C -- flock $L docker run --rm --gpus all --ipc=host --name c49rw999-enum-lsa --entrypoint "" -e VLLM_DS41_LSA=1 -w $V -v $B/src/served_topk_configs.py:/tmp/served_topk_configs.py:ro -v $HFC:/tmp/hf_config.json:ro -v $J:/out $C bash -lc "python3 /tmp/served_topk_configs.py /tmp/hf_config.json /out/served-topk-configs-lsa1.json"
# 2. Paired timings of the served configurations bench_2boot.py's grid does not time, two boots
for boot in 1 2; do
  $B/run.sh bench-served-gaps-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49rw999-gaps-b$boot --entrypoint "" -w $V -v $B/src/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/src/bench_served_gaps.py:/tmp/bench_served_gaps.py:ro -v $B/src/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro -v $J:/out $C bash -lc "python3 /tmp/bench_served_gaps.py /out/bench-served-gaps-boot$boot.json"
done
