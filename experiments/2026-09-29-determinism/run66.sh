#!/bin/bash
# usage: run66.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    mhc-mt2 overlay on dgx1)
# Cluster stopped: mhc_capture_replay.py on the run56 captures with MHC_REPLAY_SET=sweep (overlay mhc-mt2 and
# the vLLM clone's b12x_layers.py): native 16 rows per CTA with 9, 13 and 25 partial sums per CTA, and the
# TF32 projection with 40 K slices under five pairs of launch geometries (graph-size plans / capacity plan),
# each checked bit for bit against its baseline at 1-48 and 1024 rows, row invariance per configuration,
# timings. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/mhc6
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
W=$HOME/work/vllm-gemv
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
sha256sum $HOME/spark3-overlay/mhc-mt2/b12x/norm/mhc/*.py $W/vllm/models/deepseek_v4_1/b12x_layers.py \
  > "$out/overlay-bytes-run66.txt"
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py $E/mhc_capture_replay.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
M=$HOME/spark3-overlay/mhc-mt2/b12x/norm/mhc
MB=/opt/spark3/candidate/b12x/b12x/norm/mhc
log "mhc_capture_replay sweep"
docker run --rm --gpus all --ipc=host -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 \
  -e B12X_W4A8_TINY_DECODE=0 -e B12X_AUTOTUNE=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute \
  -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x -e MHC_REPLAY_SET=sweep \
  -v $W/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $M/_kernels.py:$MB/_kernels.py:ro -v $M/_preparation.py:$MB/_preparation.py:ro -v $M/_tuning.py:$MB/_tuning.py:ro \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -v $PWD/results/private/determinism/mhc/captures:/cap:ro \
  -w /opt/spark3/candidate/b12x --entrypoint python3 $IMAGE /r/mhc_capture_replay.py /cap \
  > "$out/mhc_capture_replay.txt" 2>&1
log "mhc_capture_replay exit $?"
grep -hE '"groups"|"bits"|done|Error|Traceback' "$out/mhc_capture_replay.txt" | cut -c1-260 | head -80
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
