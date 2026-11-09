. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
for boot in 1 2; do
  # README verification: GPU unit tests (sp_project_reduce_scatter + sp_prefill)
  $B/run.sh 979 gpu-pytest-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-979-gpu-b$boot --entrypoint pytest -w /tmp $C --noconftest -v $V/tests/models/test_deepseek_v41_sp_project_reduce_scatter.py $V/tests/models/test_deepseek_v4_1_sp_prefill.py
  # README verification: CPU unit tests (sp_prefill + engram), no GPU in the container
  $B/run.sh 979 cpu-pytest-candidate49-boot$boot $C -- flock $L docker run --rm --name c49om-979-cpu-b$boot --entrypoint pytest -w /tmp $C --noconftest -v $V/tests/models/test_deepseek_v4_1_sp_prefill.py $V/tests/models/test_deepseek_v4_1_engram.py
done
