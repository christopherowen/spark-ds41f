. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
EXP=$B/src/pr-999/experiments/2026-11-08-long-context-indexer-topk
J=$B/out/pr-999/json; mkdir -p $J
# README bench verbatim: /tmp_host/indexer_baseline.py = 1107 indexer.py, unmodified (expected to fail to import on the candidate)
$B/run.sh 999 bench-readme-unshimmed-candidate49 $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-999-bench-unshim --entrypoint "" -w $V -v $EXP/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/ref/1107/indexer.py:/tmp_host/indexer_baseline.py:ro -v $J:/out $C bash -lc "python3 /tmp/bench_2boot.py /out/bench-readme-unshimmed-candidate49.json"
for boot in 1 2; do
  # README step 1: pytest suite + indexer reads check (candidate image carries 0762..0767, so no copy-in)
  $B/run.sh 999 pytest-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-999-pyt-b$boot --entrypoint "" -w $V $C bash -lc "pytest --noconftest -q tests/kernels/attention/test_deepseek_v41_tilelang.py tests/models/test_deepseek_v4_1_lsa.py tests/models/test_deepseek_v4_1_attention.py; a=\$?; python3 tests/kernels/attention/deepseek_v41_indexer_reads.py; b=\$?; echo pytest_rc=\$a reads_rc=\$b; exit \$((a|b))"
  # README step 2: paired bench; baseline = 1107 indexer.py with the _e2m1_code shim (shim/indexer_baseline_1107.diff)
  $B/run.sh 999 bench-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-999-bench-cand-b$boot --entrypoint "" -w $V -v $EXP/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/shim/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro -v $J:/out $C bash -lc "python3 /tmp/bench_2boot.py /out/bench-candidate49-boot$boot.json"
  # same command on the 1107 release image (base and cand modules are then both 1107: an A/A, if it runs)
  $B/run.sh 999 bench-1107-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49om-999-bench-1107-b$boot --entrypoint "" -w $V -v $EXP/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/shim/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro -v $J:/out $R bash -lc "python3 /tmp/bench_2boot.py /out/bench-1107-boot$boot.json"
done
