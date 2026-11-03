#!/bin/bash
# Run test files of /tmp/gate-tests-fix/vllm (mounted over the image's copies) in spark-ds41f:candidate-38,
# as the release gate does (scripts/lab.py kernel-local: same env, seccomp, cache). usage: run_tests.sh LABEL PYTEST_ARGS...
set -u
label=$1; shift
tree=/tmp/gate-tests-fix/vllm
root=/opt/spark-ds41f/candidate
mounts=()
for f in tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py tests/models/test_deepseek_v4_1_engram.py tests/v1/attention/test_b12x_v41_ced_metadata.py tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_worker.py tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py; do
  mounts+=(-v "$tree/$f:$root/vllm/$f:ro")
done
mkdir -p /tmp/gate-tests-fix/out/$label
exec flock /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host --ulimit=core=0 \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/b -e TILELANG_CACHE_DIR=/c/tilelang-release-tests \
  -e ENGRAM_ROWS_TEST_DIR=/c/engram-rows-test -e DIRECT_IO_TEST_DIR=/c/direct-io-test \
  --security-opt=seccomp=unconfined --ulimit=memlock=-1:-1 \
  "${mounts[@]}" -v /tmp/gate-tests-fix/rt:/b:ro -v $HOME/.cache/spark-lab/compile:/c \
  -v /tmp/gate-tests-fix/out/$label:/out \
  -w $root/vllm --entrypoint python3 spark-ds41f:candidate-38 \
  -m pytest -q -rfEs -p no:cacheprovider -p tblib_stub -p offline_skip -o junit_family=xunit1 \
  --junitxml /out/junit.xml --basetemp /c/pytest-tmp-gate-fix/$label "$@"
