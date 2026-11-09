. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
# Re-run of pr-979.sh without --noconftest: the README gives no pytest command, and with --noconftest the
# GPU run lacks the default_vllm_config fixture (12 setup errors) and the CPU run fails to collect the engram
# file (tilelang stub). The PR's receipts (396 GPU / 219+17 CPU) were evidently collected with tests/conftest.py.
for boot in 1 2; do
  $B/run.sh 979 gpu-pytest-conftest-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-979-gpuc-b$boot --entrypoint pytest -w /tmp $C -v $V/tests/models/test_deepseek_v41_sp_project_reduce_scatter.py $V/tests/models/test_deepseek_v4_1_sp_prefill.py
  $B/run.sh 979 cpu-pytest-conftest-candidate49-boot$boot $C -- flock $L docker run --rm --name c49om-979-cpuc-b$boot --entrypoint pytest -w /tmp $C -v $V/tests/models/test_deepseek_v4_1_sp_prefill.py $V/tests/models/test_deepseek_v4_1_engram.py
done
