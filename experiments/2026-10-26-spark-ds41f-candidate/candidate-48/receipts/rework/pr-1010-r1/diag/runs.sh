#!/bin/bash
cd ~/spark-lab/agents/candidate-48-rework-1010
D=$PWD
V=/opt/spark-ds41f/candidate/vllm
IMG=sha256:d6ffd842978849651e756ebd39e9bf2da7ddb03e746113013533029878485a24
MNT="-v $D/final/vllm/v1/core/kv_cache_coordinator.py:$V/vllm/v1/core/kv_cache_coordinator.py:ro -v $D/final/vllm/v1/core/single_type_kv_cache_manager.py:$V/vllm/v1/core/single_type_kv_cache_manager.py:ro -v $D/final/tests/v1/core/test_prefix_cache_retention_probe.py:$V/tests/v1/core/test_prefix_cache_retention_probe.py:ro"
PYT="-e PYTHONPATH=$V:/opt/spark-ds41f/candidate/b12x:/opt/release_tests -v $D/src/release_tests:/opt/release_tests:ro"
LIST="tests/v1/core/test_prefix_cache_retention_probe.py tests/v1/core/test_prefix_group_validity.py tests/v1/core/prefix_cache/test_mamba_boundary_capture.py tests/v1/core/prefix_cache/test_mamba_boundary_capture_digest.py tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py tests/v1/core/prefix_cache/test_sparse_event_context.py tests/v1/core/test_async_scheduler.py tests/v1/core/test_boundary_admission.py tests/v1/core/test_boundary_admission_aged_cache.py tests/v1/core/test_boundary_admission_continuation.py tests/v1/core/test_boundary_admission_horizon.py tests/v1/core/test_boundary_admission_review.py tests/v1/core/test_partial_attention_rewind.py tests/v1/core/test_prefix_caching.py tests/v1/core/test_single_type_kv_cache_manager.py tests/v1/core/test_sink_draft_zero.py tests/v1/core/test_swa_inflight_window_free.py tests/v1/worker/test_boundary_checkpoint_attention.py tests/v1/worker/test_boundary_checkpoint_restore_scalars.py tests/kernels/attention/test_deepseek_v41_ced_replay.py tests/v1/attention/test_b12x_v41_ced_metadata.py"
if [ -s ~/spark-hold.json ]; then echo "hold file present, stopping"; cat ~/spark-hold.json; exit 3; fi
./run.sh pytest-pr-file-precheck -- flock /tmp/spark-lab-gpu.lock docker run --rm --name c48rw-1010-pyt-pre --gpus all --ipc=host --entrypoint pytest -w $V $MNT $PYT $IMG -p tblib_stub -q tests/v1/core/test_prefix_cache_retention_probe.py
echo "precheck exit $?"
tail -3 out/pytest-pr-file-precheck.txt
for b in 1 2; do
  ./run.sh probe-rework-boot$b -- flock /tmp/spark-lab-gpu.lock docker run --rm --name c48rw-1010-probe-b$b --gpus all --ipc=host --entrypoint python3 -w $V $MNT -v $D/src/probe:/probe:ro -v $D/out:/out $IMG /probe/probe_prefix_cache_retention.py --boot $b --output /out/probe-rework-boot$b.json
  echo "probe boot$b exit $?"
done
for b in 1 2; do
  ./run.sh pytest-receipt-list-rework-boot$b -- flock /tmp/spark-lab-gpu.lock docker run --rm --name c48rw-1010-pyt-b$b --gpus all --ipc=host --entrypoint pytest -w $V $MNT $PYT $IMG -p tblib_stub $LIST
  echo "pytest boot$b exit $?"; tail -1 out/pytest-receipt-list-rework-boot$b.txt
done
echo ALLDONE
