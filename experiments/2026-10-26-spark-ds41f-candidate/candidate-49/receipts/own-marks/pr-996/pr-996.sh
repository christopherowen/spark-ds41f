. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
J=$B/out/pr-996/json; mkdir -p $J
for boot in 1 2; do
  # README pytest (patched-file mounts dropped: the image carries 0767)
  $B/run.sh 996 pytest-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-996-pyt-b$boot --entrypoint "" -w $V $C pytest --noconftest tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py
  # route_local bench: the PR ships none; our harness (shim/bench_996_route_local.py) on the candidate and on 1107
  $B/run.sh 996 route-local-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-996-rl-cand-b$boot --entrypoint "" -w $V -v $B/shim/bench_996_route_local.py:/tmp/bench_996_route_local.py:ro -v $J:/out $C python3 /tmp/bench_996_route_local.py bench /out/route-local-candidate49-boot$boot.json
  $B/run.sh 996 route-local-1107-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49om-996-rl-1107-b$boot --entrypoint "" -w $V -v $B/shim/bench_996_route_local.py:/tmp/bench_996_route_local.py:ro -v $J:/out $R python3 /tmp/bench_996_route_local.py bench /out/route-local-1107-boot$boot.json
  # supplementary: the same route_local calls under torch.inference_mode, as the vLLM worker runs them
  $B/run.sh 996 inference-mode-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-996-im-cand-b$boot --entrypoint "" -w $V -v $B/shim/bench_996_route_local.py:/tmp/bench_996_route_local.py:ro -v $J:/out $C python3 /tmp/bench_996_route_local.py infer /out/inference-mode-candidate49-boot$boot.json
  $B/run.sh 996 inference-mode-1107-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49om-996-im-1107-b$boot --entrypoint "" -w $V -v $B/shim/bench_996_route_local.py:/tmp/bench_996_route_local.py:ro -v $J:/out $R python3 /tmp/bench_996_route_local.py infer /out/inference-mode-1107-boot$boot.json
done
