. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
EXP=$B/src/pr-1013/experiments/2026-11-08-completions-grammar-and-prefill-lanes
W=$B/work/pr-1013; mkdir -p $W; chmod 777 $W
FILES="tests/v1/structured_output/test_structured_output_manager.py tests/v1/spec_decode/test_mtp_structured_output.py tests/v1/core/test_prefill_lane_livelock.py tests/v1/core/test_prefill_qos_lanes.py tests/v1/core/test_aligned_prefill_lanes.py tests/v1/core/test_prefill_yield.py tests/v1/core/test_aligned_prefill_chunks.py tests/v1/core/test_prefill_compute_share_scheduler.py"
BENCH_FLAGS="--gpus=all --security-opt=seccomp=unconfined --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 -e HF_HUB_OFFLINE=1"
for BOOT in 1 2; do
  # README step 1 on the candidate image (patched files already in the image; tblib_stub from scripts/release_tests)
  $B/run.sh 1013 pytest-candidate48-boot$BOOT $C -- flock $L docker run --rm --name c48om-1013-pyt-b$BOOT --gpus=all \
    -e HF_HUB_OFFLINE=1 -v /home/chris/.cache/huggingface:/root/.cache/huggingface:ro -v $RT:/opt/release_tests:ro \
    --entrypoint bash $C -c "set -e; cd /opt/spark-ds41f/candidate/vllm; PYTHONPATH=/opt/release_tests:\$PYTHONPATH pytest -o addopts='' -p tblib_stub -q $FILES"
  # README step 2 on the candidate image
  $B/run.sh 1013 bench-candidate48-boot$BOOT $C -- flock $L docker run --rm --name c48om-1013-bench-cand-b$BOOT $BENCH_FLAGS \
    -v /home/chris/.cache/huggingface/hub:/mnt/ds41-hub:ro -v $W:/mnt/host-tmp:rw -v $EXP/bench_suite.py:/mnt/host-tmp/bench_suite.py:ro \
    --entrypoint bash $C -c "set -e; cd /opt/spark-ds41f/candidate/vllm; python3 /mnt/host-tmp/bench_suite.py boot$BOOT"
  # README step 2 verbatim on the 1107 release image
  $B/run.sh 1013 bench-1107-boot$BOOT $R -- flock $L docker run --rm --name c48om-1013-bench-1107-b$BOOT $BENCH_FLAGS \
    -v /home/chris/.cache/huggingface/hub:/mnt/ds41-hub:ro -v $W:/mnt/host-tmp:rw -v $EXP/bench_suite.py:/mnt/host-tmp/bench_suite.py:ro \
    --entrypoint bash $R -c "set -e; cd /opt/spark-ds41f/candidate/vllm; python3 /mnt/host-tmp/bench_suite.py 1107-boot$BOOT"
  # Adapted run on the 1107 release image (shim drops 0766's two SchedulerConfig kwargs)
  $B/run.sh 1013 bench-1107-shim-boot$BOOT $R -- flock $L docker run --rm --name c48om-1013-bench-1107s-b$BOOT $BENCH_FLAGS \
    -v /home/chris/.cache/huggingface/hub:/mnt/ds41-hub:ro -v $W:/mnt/host-tmp:rw -v $EXP/bench_suite.py:/mnt/host-tmp/bench_suite.py:ro \
    -v $B/shim/run_bench_suite_on_1107.py:/mnt/shim/run_bench_suite_on_1107.py:ro \
    --entrypoint bash $R -c "set -e; cd /opt/spark-ds41f/candidate/vllm; python3 /mnt/shim/run_bench_suite_on_1107.py /mnt/host-tmp/bench_suite.py 1107-shim-boot$BOOT"
done
