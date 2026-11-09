#!/bin/bash
# #987 (WO-B part): our WO-B bench h987.py (the PR's bench_wo_b.py is not committed),
# ABBA in one session (base, c51, c51, base), then the mark's tests on c51, two boots.
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh; mkdir -p $B/out/pr-987/json
for spec in "base 1" "c51 1" "c51 2" "base 2"; do set -- $spec
  $OM 987 wob-$1-boot$2 $1 -- python3 /h/h987.py /out/json/wob-$1-boot$2.json
done
for k in 1 2; do
  $OM 987 pytest-c51-boot$k c51 -- pytest -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_wo_b_wide_decode_128_rows_matches_prefill \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_mxfp8_decode_tile_repeats \
    tests/kernels/quantization/test_deepseek_v41_block_scale_words.py
done
