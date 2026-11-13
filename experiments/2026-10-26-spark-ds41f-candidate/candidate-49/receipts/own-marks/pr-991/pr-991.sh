. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
EXP=$B/src/pr-991/experiments/2026-11-08-hw-e2m1x2-indexer-gemm-quant
J=$B/out/pr-991/json; mkdir -p $J
K='test_lsa or test_block_scale_words or test_mxfp8_gemm_rope_mxfp4_decode_matches_reference or test_hw_e2m1x2_codegen_in_gemm_and_indexer_quantizers or test_mxfp4_quantize or test_write_index_keys or test_norm_rope_write_index_keys or test_indexer_dense or test_indexer_candidates'
for boot in 1 2; do
  # README section 4 pytest (patched-file mounts dropped: the image carries 0764)
  $B/run.sh 991 pytest-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-991-pyt-b$boot --entrypoint pytest -v /usr/local/cuda-13.0/bin/nvdisasm:/usr/local/bin/nvdisasm:ro -w $V $C --noconftest -o addopts= -o cache_dir=/tmp/pytest_cache -v -k "$K" tests/models/test_deepseek_v4_1_lsa.py tests/kernels/quantization/test_deepseek_v41_block_scale_words.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py tests/kernels/attention/test_deepseek_v41_tilelang.py
  # README section 5 bench (defaults: 80 rounds x 25 repeats)
  $B/run.sh 991 bench-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-991-bench-cand-b$boot --entrypoint "" -v /usr/local/cuda-13.0/bin/nvdisasm:/usr/local/bin/nvdisasm:ro -v $EXP/bench_hw_e2m1x2_quant.py:/tmp/bench_hw_e2m1x2_quant.py:ro -v $J:/out -w $V $C bash -lc "python3 /tmp/bench_hw_e2m1x2_quant.py --out /out/bench-candidate49-boot$boot.json"
done
# the same bench on 1107 (it imports _E2M1_HEADER, which 1107 lacks; recorded once)
$B/run.sh 991 bench-1107-boot1 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49om-991-bench-1107-b1 --entrypoint "" -v /usr/local/cuda-13.0/bin/nvdisasm:/usr/local/bin/nvdisasm:ro -v $EXP/bench_hw_e2m1x2_quant.py:/tmp/bench_hw_e2m1x2_quant.py:ro -v $J:/out -w $V $R bash -lc "python3 /tmp/bench_hw_e2m1x2_quant.py --out /out/bench-1107-boot1.json"
