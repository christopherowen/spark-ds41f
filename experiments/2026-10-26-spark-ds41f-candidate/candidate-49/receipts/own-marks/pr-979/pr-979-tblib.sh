. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
# Second re-run: with tests/conftest.py loaded, the image lacks tblib (exit 4), so the repository's
# scripts/release_tests/tblib_stub.py (main 51be7406, the release-test runner's stand-in) is mounted read-only
# and loaded with -p tblib_stub, as candidate-48's own-mark checks did.
RT=$B/spark-ds41f/scripts/release_tests
PP=$V:/opt/spark-ds41f/candidate/b12x:/opt/release_tests
for boot in 1 2; do
  $B/run.sh 979 gpu-pytest-tblibstub-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-979-gput-b$boot --entrypoint pytest -w /tmp -e PYTHONPATH=$PP -v $RT:/opt/release_tests:ro $C -p tblib_stub -v $V/tests/models/test_deepseek_v41_sp_project_reduce_scatter.py $V/tests/models/test_deepseek_v4_1_sp_prefill.py
  $B/run.sh 979 cpu-pytest-tblibstub-candidate49-boot$boot $C -- flock $L docker run --rm --name c49om-979-cput-b$boot --entrypoint pytest -w /tmp -e PYTHONPATH=$PP -v $RT:/opt/release_tests:ro $C -p tblib_stub -v $V/tests/models/test_deepseek_v4_1_sp_prefill.py $V/tests/models/test_deepseek_v4_1_engram.py
done
