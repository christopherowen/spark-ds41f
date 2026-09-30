#!/bin/bash
# usage: run12.sh   (on dgx1, deployment checkout at this experiment's commit)
# With the cluster stopped: linear_poison_check.py on dgx3 in a fresh r5m
# container (turbo off and on), then compute-sanitizer initcheck on the down
# projection at 6 rows. Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
scp -q $E/linear_poison_check.py dgx3:/tmp/linear_poison_check.py
log "linear poison checks on dgx3"
ssh dgx3 'bash -s' > "$out/linear-poison.txt" 2>&1 <<'REMOTE'
B=/opt/spark3/candidate/b12x
run() {
  docker run --rm --gpus all --ipc=host -w $B \
    -v /tmp/linear_poison_check.py:/tmp/linear_poison_check.py:ro \
    -v /usr/local/cuda-13.0/compute-sanitizer:/opt/compute-sanitizer:ro \
    -e CUTE_DSL_ARCH=sm_121a -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute \
    -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc "$@"
}
echo "== turbo off"
run -e B12X_DENSE_SPLITK_TURBO=0 --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/linear_poison_check.py
echo "== turbo on"
run -e B12X_DENSE_SPLITK_TURBO=1 --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/linear_poison_check.py
echo "== initcheck, 6 rows"
run -e B12X_DENSE_SPLITK_TURBO=0 -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 --entrypoint /opt/compute-sanitizer/compute-sanitizer vllm-ds41f-kkref:04c30fa98e79-r5m --tool initcheck --print-limit 10 python3 /tmp/linear_poison_check.py --quick 2>&1 | tail -40
REMOTE
log "linear poison checks done"
grep -E "^==|rows|OK|FAIL|ERROR SUMMARY|Uninitialized|Error" "$out/linear-poison.txt" | head -60
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
