#!/bin/bash
# q.sh: the candidate-53 own-mark queue on sp8, one GPU job at a time (each through run.sh,
# which takes /tmp/spark-lab-gpu.lock and refuses on a hold file or a running container).
# Per PR: the PR's own command(s) as recorded, then the image A/B, ABBA:
# base boot1, c53 boot1, c53 boot2, base boot2. A job refused for a running container is
# retried every 60 s for up to 30 min; a hold file stops the queue.
B=$HOME/spark-lab/agents/candidate-53-own-marks; cd $B
H=$B/harness; X=$B/src; BUN=$X/experiments/2026-10-26-spark-ds41f-candidate/candidate-53/bundles/release-tests
E960=$X/pr-960/experiments/2026-11-08-cudagraph-decode-optimizations
E971=$X/pr-971/experiments/2026-11-08-copy-drafter-survival-and-tile-cap
E977=$X/pr-977/experiments/2026-11-08-copy-drafter-long-context-scan
E984=$X/pr-984/experiments/2026-11-08-uma-allocator-hygiene
E998=$X/pr-998/experiments/2026-11-08-display-carveout-weight-expansion
r() {  # r PR LABEL ARM ...
  for i in $(seq 1 30); do
    [ -e $HOME/spark-hold.json ] && { echo "HOLD present; queue stops at $2"; exit 9; }
    common/run.sh "$@"; rc=$?
    [ $rc -ne 3 ] && return $rc
    sleep 60
  done
}
step=${1:-all}
if [ $step = all ] || [ $step = 960 ]; then
  M="-v $E960:/exp:ro -v $H:/h:ro"
  r 960 script-base-boot1 base $M -- python3 /h/warm.py /exp/screen_cudagraph_wins.py --boot 1 --output /out/script-base-boot1.json
  r 960 script-c53-boot1 c53 $M -- python3 /h/warm.py /exp/screen_cudagraph_wins.py --boot 1 --output /out/script-c53-boot1.json
  r 960 script-c53-boot2 c53 $M -- python3 /h/warm.py /exp/screen_cudagraph_wins.py --boot 2 --output /out/script-c53-boot2.json
  r 960 ab-base-boot1 base $M -- python3 /h/ab960.py --label base-boot1 --out /out/ab-base-boot1.jsonl
  r 960 ab-c53-boot1 c53 $M -- python3 /h/ab960.py --label c53-boot1 --out /out/ab-c53-boot1.jsonl
  r 960 ab-c53-boot2 c53 $M -- python3 /h/ab960.py --label c53-boot2 --out /out/ab-c53-boot2.jsonl
  r 960 ab-base-boot2 base $M -- python3 /h/ab960.py --label base-boot2 --out /out/ab-base-boot2.jsonl
  for k in 1 2; do r 960 pytest-c53-boot$k c53 -v $BUN:/bundle:ro -e PYTHONPATH=/bundle -- pytest -p tblib_stub -v tests/v1/attention/test_b12x_v41_ced_metadata.py tests/v1/cudagraph/test_cudagraph_manager.py; done
fi
if [ $step = all ] || [ $step = 971 ]; then
  M="-v $E971/screen_copy_drafter_smooth.py:/tmp/screen_copy_drafter_smooth.py:ro -v $H:/h:ro"
  r 971 script-base-boot1 base $M -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot1
  r 971 script-c53-boot1 c53 $M -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot1
  r 971 script-c53-boot2 c53 $M -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot2
  T="-e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2"
  r 971 ab-base-boot1 base $M $T -- python3 /h/warm.py /h/ab971.py --label base-boot1 --out /out/ab-base-boot1.jsonl
  r 971 ab-c53-boot1 c53 $M $T -- python3 /h/warm.py /h/ab971.py --label c53-boot1 --out /out/ab-c53-boot1.jsonl
  r 971 ab-c53-boot2 c53 $M $T -- python3 /h/warm.py /h/ab971.py --label c53-boot2 --out /out/ab-c53-boot2.jsonl
  r 971 ab-base-boot2 base $M $T -- python3 /h/warm.py /h/ab971.py --label base-boot2 --out /out/ab-base-boot2.jsonl
  for k in 1 2; do r 971 pytest-c53-boot$k c53 -- pytest -v --noconftest tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py tests/v1/spec_decode/test_adaptive_verification.py tests/v1/worker/test_prepare_kernels.py; done
fi
if [ $step = all ] || [ $step = 977 ]; then
  M="-v $E977:/exp:ro -v $H:/h:ro"
  r 977 bench-base-boot1 base $M -- python3 /h/warm.py /exp/bench_copy_match_long_context.py --label base --out /out/base-boot1.jsonl
  r 977 bench-c53-boot1 c53 $M -- python3 /h/warm.py /exp/bench_copy_match_long_context.py --label candidate --out /out/candidate-boot1.jsonl
  r 977 bench-c53-boot2 c53 $M -- python3 /h/warm.py /exp/bench_copy_match_long_context.py --label candidate --out /out/candidate-boot2.jsonl
  r 977 bench-base-boot2 base $M -- python3 /h/warm.py /exp/bench_copy_match_long_context.py --label base --out /out/base-boot2.jsonl
  r 977 ids-base-boot1 base $M -- python3 /h/ids977.py --label base-boot1 --out /out/ids-base-boot1.jsonl
  r 977 ids-c53-boot1 c53 $M -- python3 /h/ids977.py --label c53-boot1 --out /out/ids-c53-boot1.jsonl
  r 977 ids-c53-boot2 c53 $M -- python3 /h/ids977.py --label c53-boot2 --out /out/ids-c53-boot2.jsonl
  r 977 ids-base-boot2 base $M -- python3 /h/ids977.py --label base-boot2 --out /out/ids-base-boot2.jsonl
  r 977 gate-base-boot1 base -- python3 benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy
  r 977 gate-c53-boot1 c53 -- python3 benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy
  r 977 gate-c53-boot2 c53 -- python3 benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy
  r 977 gate-base-boot2 base -- python3 benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy
  for k in 1 2; do r 977 pytest-c53-boot$k c53 -- pytest -o addopts= --noconftest -v tests/v1/spec_decode/test_copy_drafts.py; done
fi
if [ $step = all ] || [ $step = 984 ]; then
  M="-v $E984/bench_uma_allocator_hygiene.py:/tmp/bench_uma_allocator_hygiene.py:ro -v $H:/h:ro --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm"
  r 984 bench-base-boot1 base $M -- python3 /tmp/bench_uma_allocator_hygiene.py --boot-label boot1 --output /out/bench-base-boot1.json
  r 984 bench-c53-boot1 c53 $M -- python3 /tmp/bench_uma_allocator_hygiene.py --boot-label boot1 --output /out/bench-c53-boot1.json
  r 984 bench-c53-boot2 c53 $M -- python3 /tmp/bench_uma_allocator_hygiene.py --boot-label boot2 --output /out/bench-c53-boot2.json
  r 984 bench-base-boot2 base $M -- python3 /tmp/bench_uma_allocator_hygiene.py --boot-label boot2 --output /out/bench-base-boot2.json
  r 984 flush-c53-boot1 c53 -v $H:/h:ro -- python3 /h/flush984.py c53-boot1 /out/flush-c53-boot1.json
  r 984 flush-c53-boot2 c53 -v $H:/h:ro -- python3 /h/flush984.py c53-boot2 /out/flush-c53-boot2.json
  for k in 1 2; do r 984 pytest-c53-boot$k c53 --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v $B/stub_pkgs:/tmp/stub_pkgs:ro -e PYTHONPATH=/tmp/stub_pkgs:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/vllm/tests/vllm_test_utils -- pytest -v -k "not test_memory_profiling" tests/utils_/test_mem_utils.py tests/utils_/test_gc_utils.py tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_worker.py tests/model_executor/model_loader/test_direct_loader.py tests/model_executor/model_loader/test_direct_loader_cuda.py; done
fi
if [ $step = all ] || [ $step = 998 ]; then
  M="-v $E998/bench_carveout_expansion.py:/tmp/run_carveout_bench.py:ro -v $H:/h:ro --device /dev/dri/card0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  r 998 script-base-boot1 base $M -- python3 /h/warm.py /tmp/run_carveout_bench.py 1
  r 998 script-c53-boot1 c53 $M -- python3 /h/warm.py /tmp/run_carveout_bench.py 1
  r 998 script-c53-boot2 c53 $M -- python3 /h/warm.py /tmp/run_carveout_bench.py 2
  T="-e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1"
  r 998 ab-base-boot1 base $M $T -- python3 /h/warm.py /h/ab998.py base-boot1 /out/ab-base-boot1.json
  r 998 ab-c53-boot1 c53 $M $T -- python3 /h/warm.py /h/ab998.py c53-boot1 /out/ab-c53-boot1.json
  r 998 ab-c53-boot2 c53 $M $T -- python3 /h/warm.py /h/ab998.py c53-boot2 /out/ab-c53-boot2.json
  r 998 ab-base-boot2 base $M $T -- python3 /h/warm.py /h/ab998.py base-boot2 /out/ab-base-boot2.json
  P='import sys, types, pytest
tblib = types.ModuleType("tblib")
ps = types.ModuleType("tblib.pickling_support")
ps.install = lambda *a, **kw: None
tblib.pickling_support = ps
sys.modules["tblib"] = tblib
sys.modules["tblib.pickling_support"] = ps
sys.exit(pytest.main([
    "-q",
    "/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_display_carveout.py",
    "/opt/spark-ds41f/candidate/vllm/tests/models/test_dspark_mla.py",
    "/opt/spark-ds41f/candidate/vllm/tests/model_executor/model_loader/test_direct_loader.py",
    "/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_gpu_model_runner_v2.py",
    "/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_embedding.py",
]))'
  for k in 1 2; do r 998 pytest-c53-boot$k c53 --device /dev/dri/card0 -- python3 -c "$P"; done
fi
echo "QUEUE DONE $step $(date -u +%FT%TZ)"
