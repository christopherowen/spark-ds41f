#!/bin/bash
# #987 supplement: a second ABBA pair of boots of the WO-B bench (h987b.py: h987's cells,
# plus kernel time with a sync after each call and WO-A alone), then the mark's tests
# again with the release bundle's tblib stand-in (the image lacks tblib, which
# tests/conftest.py imports; the first pytest runs stopped there, exit 4).
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh; mkdir -p $B/out/pr-987/json
for spec in "base 3" "c51 3" "c51 4" "base 4"; do set -- $spec
  $OM 987 wobb-$1-boot$2 $1 -- python3 /h/h987b.py /out/json/wobb-$1-boot$2.json
done
for k in 1 2; do
  $OM 987 pytest-stub-c51-boot$k c51 -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h -- pytest -p tblib_stub -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_wo_b_wide_decode_128_rows_matches_prefill \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_mxfp8_decode_tile_repeats \
    tests/kernels/quantization/test_deepseek_v41_block_scale_words.py
done
