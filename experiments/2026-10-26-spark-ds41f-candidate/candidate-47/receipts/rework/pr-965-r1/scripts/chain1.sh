#!/bin/bash
# Diagnosis, the corrected stack's tests, then the step harness on two boots per stack.
cd ~/spark-lab/agents/candidate-47-rework-965
./run.sh diag-c47 c47 python3 /harness/diag_declared.py
./run.sh diag-fix fix python3 /harness/diag_declared.py
./run.sh fix-tilelang-inputs fix pytest -p tblib_stub -q -rf tests/v1/worker/test_gpu_tilelang_inputs.py
./run.sh fix-release-subset fix python3 /opt/release_tests/release_test_runner.py /harness/manifest-worker-spec-decode.json /out/release-subset-fix
for b in 1 2; do
  for o in c47 fix; do
    ./run.sh harness-$o-b$b $o bash -c "python3 /harness/bench_step_fastpath.py --boot $b --output /out/harness-$o-b$b.json && python3 /harness/bench_clean_apply.py --label $o-b$b --output /out/clean-apply-$o-b$b.json"
  done
done
echo chain1-done
