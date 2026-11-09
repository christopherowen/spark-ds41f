. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
EXP=$B/src/pr-972/experiments/2026-11-08-sse-keepalive-heartbeat
for boot in 1 2; do
  # README step 1 (pytest); the candidate image already carries the patched files, so no copy-in
  $B/run.sh 972 pytest-candidate48-boot$boot $C -- flock $L docker run --rm --name c48om-972-pyt-b$boot --entrypoint "" $C bash -lc "set -euo pipefail; cd /opt/spark-ds41f/candidate/vllm; pytest --noconftest -v tests/entrypoints/serve/utils/test_api_cancellation.py tests/v1/engine/test_request_submission_cleanup.py tests/v1/engine/test_utility_call_cleanup.py"
  # README step 2 (bench), candidate image then the 1107 release image as the baseline
  $B/run.sh 972 bench-candidate48-boot$boot $C -- flock $L docker run --rm --name c48om-972-bench-cand-b$boot --entrypoint "" -v $EXP/bench_sse_keepalive.py:/tmp/bench_sse_keepalive.py:ro $C bash -lc "set -euo pipefail; python3 /tmp/bench_sse_keepalive.py"
  $B/run.sh 972 bench-1107-boot$boot $R -- flock $L docker run --rm --name c48om-972-bench-1107-b$boot --entrypoint "" -v $EXP/bench_sse_keepalive.py:/tmp/bench_sse_keepalive.py:ro $R bash -lc "set -euo pipefail; python3 /tmp/bench_sse_keepalive.py"
done
