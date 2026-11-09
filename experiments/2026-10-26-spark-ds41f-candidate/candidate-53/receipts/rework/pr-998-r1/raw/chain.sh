#!/bin/bash
# Sequential runs for the #998 round-1 rework; each run takes the GPU lock on its own.
D=$HOME/spark-lab/agents/candidate-53-rework-998
cd $D
TESTS="tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py tests/v1/worker/test_display_carveout.py tests/models/test_dspark_mla.py tests/model_executor/model_loader/test_direct_loader.py tests/v1/worker/test_gpu_model_runner_v2.py tests/kernels/test_deepseek_v41_tilelang_embedding.py tests/kernels/moe/test_deepseek_v41_tilelang_moe.py"
TARGS=(-v $D/bundle/release-tests:/b:ro -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x
       --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v $D/scratch:/c/scratch
       -e DIRECT_IO_TEST_DIR=/c/scratch/direct-io-test -e ENGRAM_ROWS_TEST_DIR=/c/scratch/engram-rows-test)
PYT=(python3 -m pytest -q -rfEs -p no:cacheprovider -p tblib_stub -p offline_skip -o faulthandler_timeout=300)
BARGS=(-v $D/scripts:/s:ro -v $D/out/json:/o -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1
       -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 -e VLLM_DS41_HEAD_DTYPE=float32)
./run.sh c53 repro-c53-scale-codes-alone "${TARGS[@]}" -- "${PYT[@]}" tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py
./run.sh fix tests-fix-run1 "${TARGS[@]}" -- "${PYT[@]}" $TESTS
./run.sh fix tests-fix-run2 "${TARGS[@]}" -- "${PYT[@]}" $TESTS
./run.sh c53 bench-serving-c53-boot1 "${BARGS[@]}" -- python3 /s/bench_serving_path.py c53 boot1 /o/bench-serving-c53-boot1.json
./run.sh base bench-serving-base-boot1 "${BARGS[@]}" -- python3 /s/bench_serving_path.py base boot1 /o/bench-serving-base-boot1.json
./run.sh fix bench-serving-fix-boot1 "${BARGS[@]}" -- python3 /s/bench_serving_path.py fix boot1 /o/bench-serving-fix-boot1.json
./run.sh fix bench-serving-fix-boot2 "${BARGS[@]}" -- python3 /s/bench_serving_path.py fix boot2 /o/bench-serving-fix-boot2.json
./run.sh base bench-serving-base-boot2 "${BARGS[@]}" -- python3 /s/bench_serving_path.py base boot2 /o/bench-serving-base-boot2.json
./run.sh fix bench-pr998-script-fix-boot1 -v $D/scripts:/s:ro -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- python3 /s/bench_carveout_expansion.py 1
./run.sh fix bench-pr998-script-fix-boot2 -v $D/scripts:/s:ro -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- python3 /s/bench_carveout_expansion.py 2
echo CHAIN-DONE > $D/out/chain.done
