#!/bin/bash
# usage: run8.sh   (on dgx1, deployment checkout at this experiment's commit)
# With the cluster stopped, runs gemm_concurrency_check.py on dgx3 in a fresh r5m
# container (split-K turbo off, as in the deterministic arms, then on), then
# compute-sanitizer racecheck on the down projection. Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
scp -q $E/gemm_concurrency_check.py dgx3:/tmp/gemm_concurrency_check.py
log "gemm checks on dgx3"
ssh dgx3 'bash -s' > "$out/gemm-concurrency.txt" 2>&1 <<'REMOTE'
B=/opt/spark3/candidate/b12x
run() {
  docker run --rm --gpus all --ipc=host -w $B \
    -v /tmp/gemm_concurrency_check.py:/tmp/gemm_concurrency_check.py:ro \
    -v /usr/local/cuda-13.0/compute-sanitizer:/opt/compute-sanitizer:ro \
    -e CUTE_DSL_ARCH=sm_121a -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute \
    -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc "$@"
}
echo "== split-K turbo off"
run -e B12X_DENSE_SPLITK_TURBO=0 --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/gemm_concurrency_check.py
echo "== split-K turbo on"
run -e B12X_DENSE_SPLITK_TURBO=1 --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/gemm_concurrency_check.py
echo "== racecheck, turbo off"
run -e B12X_DENSE_SPLITK_TURBO=0 --entrypoint /opt/compute-sanitizer/compute-sanitizer vllm-ds41f-kkref:04c30fa98e79-r5m --tool racecheck --print-limit 20 python3 /tmp/gemm_concurrency_check.py --reps 2 --alone-only 2>&1 | tail -40
REMOTE
log "gemm checks done"
grep -E "^==|capacity|OK|FAIL|ERROR SUMMARY|hazard|Race" "$out/gemm-concurrency.txt" | head -40
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
