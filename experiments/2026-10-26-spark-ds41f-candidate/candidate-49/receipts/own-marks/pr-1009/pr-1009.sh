. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
EXP=$B/src/pr-1009/experiments/2026-11-08-sp-prefill-local-attention-front
J=$B/out/pr-1009/json; mkdir -p $J
# README pytest file list verbatim (two of its files are not in the tree; recorded once)
$B/run.sh 1009 pytest-readme-list-candidate49 $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-1009-pyt-readme --entrypoint pytest -w $V $C --noconftest -v tests/kernels/test_deepseek_v41_tilelang_front.py tests/models/test_deepseek_v4_1_attention.py tests/models/test_sp_prefill.py tests/models/test_sp_prefill_e2e.py
for boot in 1 2; do
  # the file list the PR's receipts/pytest_boot*.txt actually ran
  $B/run.sh 1009 pytest-receipt-list-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-1009-pyt-b$boot --entrypoint pytest -w $V $C --noconftest -v tests/kernels/test_deepseek_v41_tilelang_front.py tests/kernels/test_deepseek_v41_tilelang_columns.py tests/models/test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/models/test_deepseek_v4_1_prepare.py tests/models/test_deepseek_v41_serving_contract.py
  # README bench (defaults: 40 warmup, 200 iters)
  $B/run.sh 1009 bench-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-1009-bench-cand-b$boot --entrypoint "" -v $EXP/bench_sp_attention_front.py:/tmp/bench_sp_attention_front.py:ro -v $J:/out -w $V $C bash -lc "python3 /tmp/bench_sp_attention_front.py --json-out /out/bench-candidate49-boot$boot.json"
done
# the same bench on 1107 (it imports sp_attention_front, which 1107 lacks; recorded once)
$B/run.sh 1009 bench-1107-boot1 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49om-1009-bench-1107-b1 --entrypoint "" -v $EXP/bench_sp_attention_front.py:/tmp/bench_sp_attention_front.py:ro -v $J:/out -w $V $R bash -lc "python3 /tmp/bench_sp_attention_front.py --json-out /out/bench-1107-boot1.json"
