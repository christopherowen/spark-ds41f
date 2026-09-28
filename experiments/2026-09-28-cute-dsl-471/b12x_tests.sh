#!/bin/bash
# usage: b12x_tests.sh IMAGE LABEL   (on dgx1; runs on dgx3 with the cluster stopped)
# Runs the B12X tests that cover the DS4.1 kernels (W4A8 dynamic MoE, block-FP8,
# MXFP8 and BF16 GEMMs, v4.1 compressed sparse MLA, mHC norms, Engram) in IMAGE,
# one pytest per file, into results/private/b12x-tests/LABEL.log.
set -u
image=$1
label=$2
out=~/projects/spark3-vllm-ds41f/results/private/b12x-tests
mkdir -p "$out"
FILES="tests/moe/test_w4a8_dynamic_kernel.py tests/moe/test_w4a8_mx_tp_moe.py tests/moe/test_w4a8_reference.py
tests/gemm/test_block_fp8_linear.py tests/gemm/test_mxfp8_linear.py tests/gemm/test_bf16_gemv.py
tests/gemm/test_bf16_vocab_projection.py tests/attention/test_compressed_sparse_mla_v41.py
tests/attention/test_sparse_mla_decode_regimes.py tests/norm tests/sequence"
ssh -n dgx3 "timeout 5400 docker run --rm --gpus=all --memory=100g --entrypoint bash $image -c '
  pip install -q pytest >/dev/null 2>&1
  cd /opt/spark3/candidate/b12x
  python3 -c \"import importlib.metadata as m; print(\\\"cutlass-dsl\\\", m.version(\\\"nvidia-cutlass-dsl\\\"))\"
  for f in $(echo $FILES); do
    echo \"== \$f\"
    timeout 1200 python3 -m pytest -q -p no:cacheprovider -x \$f 2>&1 | tail -4
  done'" > "$out/$label.log" 2>&1
grep -E "^== |passed|failed|error|cutlass-dsl" "$out/$label.log"
