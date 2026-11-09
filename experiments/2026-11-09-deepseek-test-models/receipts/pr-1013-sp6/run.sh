#!/bin/bash
# Arms: release-test plugins without and with deepseek_models, on candidate-48 (1107 image + 0762..0771).
D=~/spark-lab/agents/ds41f-test-models
REPO=models--deepseek-ai--DeepSeek-V4.1-Flash-1017
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
FILES="tests/v1/structured_output/test_structured_output_manager.py tests/v1/spec_decode/test_mtp_structured_output.py tests/v1/core/test_prefill_lane_livelock.py tests/v1/core/test_prefill_qos_lanes.py tests/v1/core/test_aligned_prefill_lanes.py tests/v1/core/test_prefill_yield.py tests/v1/core/test_aligned_prefill_chunks.py tests/v1/core/test_prefill_compute_share_scheduler.py"
arm() {
  name=$1; plugins=$2
  cmd=(flock /tmp/spark-lab-gpu.lock docker run --rm --name ds41f-tm-$name --gpus=all
       -e HF_HUB_OFFLINE=1 -e DS41F_SNAPSHOT=/ds41f/$REPO/snapshots/$REV
       -v $HOME/.cache/huggingface/hub/$REPO:/ds41f/$REPO:ro -v $D/plugins:/b:ro
       -v $D/tree/vllm:/opt/spark-ds41f/candidate/vllm -v $D/out:/mnt/out
       --entrypoint bash spark-ds41f:1107 -c
       "cd /opt/spark-ds41f/candidate/vllm; PYTHONPATH=/b:\$PYTHONPATH python3 -m pytest -o addopts= -p no:cacheprovider $plugins -q -rfEs --junitxml=/mnt/out/junit-$name.xml $FILES")
  { echo "### $name $(date -u +%FT%TZ)"; echo "docker ps before: $(docker ps -q | wc -l); hold: $(ls ~/spark-hold.json 2>/dev/null || echo none)"; printf "%q " "${cmd[@]}"; echo; } >> $D/out/commands.log
  "${cmd[@]}" > $D/out/pytest-$name.txt 2>&1; rc=$?
  echo "$name rc=$rc $(date -u +%FT%TZ)" >> $D/out/commands.log
  echo "$name rc=$rc: $(grep -E "^(=+ )?[0-9]+ (passed|failed)" $D/out/pytest-$name.txt | tail -1)"
}
for a in "$@"; do case $a in
  offline) arm offline "-p tblib_stub -p offline_skip";;
  deepseek*) arm $a "-p tblib_stub -p offline_skip -p deepseek_models";;
esac; done
