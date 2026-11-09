. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
J=$B/out/pr-993/json; mkdir -p $J
for boot in 1 2; do
  # README section 4: unit and sanitizer tests (candidate image carries 0763, so no copy-in)
  $B/run.sh 993 pytest-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-993-pyt-b$boot --entrypoint "" -w $V $C bash -lc "pytest --noconftest -v tests/kernels/attention/test_deepseek_v41_tilelang.py tests/models/test_deepseek_v4_1_lsa.py"
  # P5 score bench: the PR ships none; our harness (shim/bench_993_score.py), baseline = 1107 indexer.py + _e2m1_code shim
  $B/run.sh 993 score-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-993-score-cand-b$boot --entrypoint "" -w $V -v $B/shim/bench_993_score.py:/tmp/bench_993_score.py:ro -v $B/shim/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro -v $J:/out $C bash -lc "python3 /tmp/bench_993_score.py /out/score-candidate49-boot$boot.json"
  # same harness on the 1107 release image: base and cand are both 1107 code (A/A floor)
  $B/run.sh 993 score-1107-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49om-993-score-1107-b$boot --entrypoint "" -w $V -v $B/shim/bench_993_score.py:/tmp/bench_993_score.py:ro -v $B/shim/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro -v $J:/out $R bash -lc "python3 /tmp/bench_993_score.py /out/score-1107-boot$boot.json"
done
