. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
EXP=$B/src/pr-1013/experiments/2026-11-08-completions-grammar-and-prefill-lanes
W=$B/work/pr-1013
FILES="tests/v1/structured_output/test_structured_output_manager.py tests/v1/spec_decode/test_mtp_structured_output.py tests/v1/core/test_prefill_lane_livelock.py tests/v1/core/test_prefill_qos_lanes.py tests/v1/core/test_aligned_prefill_lanes.py tests/v1/core/test_prefill_yield.py tests/v1/core/test_aligned_prefill_chunks.py tests/v1/core/test_prefill_compute_share_scheduler.py"
BENCH_FLAGS="--gpus=all --security-opt=seccomp=unconfined --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 -e HF_HUB_OFFLINE=1"
for BOOT in 1 2; do
  # README step 1 again with the release tests' offline_skip plugin: sp8 has no /tmp/hf-test-cache
  # (facebook/opt-125m and the other small test models), so a test whose only failure is a missing
  # offline Hugging Face resource is reported as skipped, naming it.
  $B/run.sh 1013 pytest-offline-skip-candidate48-boot$BOOT $C -- flock $L docker run --rm --name c48om-1013-pyt-os-b$BOOT --gpus=all \
    -e HF_HUB_OFFLINE=1 -v /home/chris/.cache/huggingface:/root/.cache/huggingface:ro -v $RT:/opt/release_tests:ro \
    --entrypoint bash $C -c "set -e; cd /opt/spark-ds41f/candidate/vllm; PYTHONPATH=/opt/release_tests:\$PYTHONPATH pytest -o addopts='' -p tblib_stub -p offline_skip -q -rs $FILES"
  # Adapted run on the 1107 release image, shim v2
  $B/run.sh 1013 bench-1107-shimv2-boot$BOOT $R -- flock $L docker run --rm --name c48om-1013-bench-1107s2-b$BOOT $BENCH_FLAGS \
    -v /home/chris/.cache/huggingface/hub:/mnt/ds41-hub:ro -v $W:/mnt/host-tmp:rw -v $EXP/bench_suite.py:/mnt/host-tmp/bench_suite.py:ro \
    -v $B/shim/run_bench_suite_on_1107_v2.py:/mnt/shim/run_bench_suite_on_1107_v2.py:ro \
    --entrypoint bash $R -c "set -e; cd /opt/spark-ds41f/candidate/vllm; python3 /mnt/shim/run_bench_suite_on_1107_v2.py /mnt/host-tmp/bench_suite.py 1107-shimv2-boot$BOOT"
done
