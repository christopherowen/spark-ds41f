#!/bin/bash
# #1001: the PR's bench_vocab_2boot.py at 03e2d903 with the changes in
# h/bench_vocab_2boot_c51.diff (the base's certified screen changed tilelang_vocab_greedy's
# layout; M 97/107/120, eager loops and output hashes added), behind a >=250 ms GPU
# warm-up (h/warm.py), ABBA in one session, then the PR's tests on c51, two boots.
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh; mkdir -p $B/out/pr-1001/json
for spec in "base 1" "c51 1" "c51 2" "base 2"; do set -- $spec
  $OM 1001 vocab-$1-boot$2 $1 -- python3 /h/warm.py /h/bench_vocab_2boot_c51.py /out/json/vocab-$1-boot$2.json
done
for k in 1 2; do
  $OM 1001 pytest-c51-boot$k c51 -- pytest -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py \
    tests/v1/worker/gpu/test_greedy_verification.py
done
