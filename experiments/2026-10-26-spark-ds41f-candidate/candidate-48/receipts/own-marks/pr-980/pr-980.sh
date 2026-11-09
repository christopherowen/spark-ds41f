. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
V=/opt/spark-ds41f/candidate/vllm
for boot in 1 2; do
  # README 3.1 on the candidate image (patched files are in the image, so no bind mounts)
  $B/run.sh 980 pytest-candidate48-boot$boot $C -- flock $L docker run --rm --name c48om-980-pyt-b$boot --gpus all --ipc=host --network=host --entrypoint pytest $C --noconftest -v $V/tests/v1/spec_decode/test_adaptive_verification.py $V/tests/v1/spec_decode/test_acceptance_estimator.py $V/tests/v1/spec_decode/test_copy_drafts.py $V/tests/v1/worker/test_dspark_greedy_markov.py $V/tests/distributed/test_roce_policy_no_fallback.py $V/tests/kernels/test_deepseek_v41_tilelang_confidence.py
done
