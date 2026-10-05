#!/bin/bash
# usage: run61.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    ref3m2 overlay on dgx1)
# Cluster stopped: gemv_geometry_replay.py on real rows (the run56 mHC captures through production mHC)
# with overlay gemv-geom over B12X bf16_gemv: each TMA prefill GEMV launch geometry against the production
# geometry, bit for bit at 1-4000 rows in both positions, and timings. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/geom
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
sha256sum $HOME/spark3-overlay/gemv-geom/b12x/gemm/bf16_gemv/*.py > "$out/overlay-bytes-run61.txt"
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py $E/gemv_geometry_replay.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
COMMON="-e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 -e B12X_AUTOTUNE=0
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x"
for script in gemv_geometry_replay; do
  log "$script"
  docker run --rm --gpus all --ipc=host $COMMON \
    -v $HOME/spark3-overlay/gemv-geom/b12x/gemm/bf16_gemv/_prefill.py:/opt/spark3/candidate/b12x/b12x/gemm/bf16_gemv/_prefill.py:ro -v $HOME/spark3-overlay/gemv-geom/b12x/gemm/bf16_gemv/_tuning.py:/opt/spark3/candidate/b12x/b12x/gemm/bf16_gemv/_tuning.py:ro -v $HOME/spark3-overlay/gemv-geom/b12x/gemm/bf16_gemv/_preparation.py:/opt/spark3/candidate/b12x/b12x/gemm/bf16_gemv/_preparation.py:ro \
    -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
    -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -v $PWD/results/private/determinism/mhc/captures:/cap:ro \
    -w /opt/spark3/candidate/b12x --entrypoint python3 $IMAGE /r/$script.py /cap > "$out/$script.txt" 2>&1
  log "$script exit $?"
  grep -hE 'bit_equal|invariant|compile_error|done|Error|Traceback' "$out/$script.txt" | cut -c1-260 | head -40
done
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
