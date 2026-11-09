. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
# The no-GPU container cannot collect test_deepseek_v4_1_engram.py (vllm/tilelang_utils gives tilelang = None, and
# vllm/v1/worker/gpu/tilelang_inputs.py dereferences it at import). (a) the same command on 1107, to see whether
# that is the release's behaviour; (b) the CPU suite with the GPU driver mounted but no visible device
# (CUDA_VISIBLE_DEVICES empty), which matches the PR receipt's "Triton ... 0 active driver(s)" line.
RT=$B/spark-ds41f/scripts/release_tests
PP=$V:/opt/spark-ds41f/candidate/b12x:/opt/release_tests
$B/run.sh 979 cpu-pytest-tblibstub-1107-boot1 $R -- flock $L docker run --rm --name c49om-979-cput-1107 --entrypoint pytest -w /tmp -e PYTHONPATH=$PP -v $RT:/opt/release_tests:ro $R -p tblib_stub -v $V/tests/models/test_deepseek_v4_1_sp_prefill.py $V/tests/models/test_deepseek_v4_1_engram.py
for boot in 1 2; do
  $B/run.sh 979 cpu-pytest-novisible-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES= --name c49om-979-cpunv-b$boot --entrypoint pytest -w /tmp -e PYTHONPATH=$PP -v $RT:/opt/release_tests:ro $C -p tblib_stub -v $V/tests/models/test_deepseek_v4_1_sp_prefill.py $V/tests/models/test_deepseek_v4_1_engram.py
done
