#!/bin/bash
# usage: run56.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref2, ref3m and mhc-capture overlays on every node)
# mHC recovery, step 1: captured inputs. detm-r5o-ref2-mhccap serves one cold ~3900-token real-text
# prompt with the capture armed (rank 0 saves pre and post_pre inputs and weights at layers 1, 20
# and 39). Cluster stopped: mhc_capture_replay.py on dgx1 with the candidate b12x_layers.py
# (vllm-0039) mounted: production, ref2, candidate and candidate-capacity plans, bits across batch
# sizes and positions, agreement with production at decode sizes, timings. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/mhc
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
CAP=/cache/kkref/mhc-capture
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find ref2 ref3m mhc-capture -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run56.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
start $E/cluster-detm-r5o-ref2-mhccap.json
docker exec dsv41-karmic-kraken sh -c "rm -rf $CAP && mkdir -p $CAP && touch $CAP/arm"
python3 $E/send_prompt.py http://10.0.1.71:8000 --tokens 3900 | tee "$out/capture-prompt.json"
log "prompt exit $?"
sleep 5
docker exec dsv41-karmic-kraken sh -c "rm -f $CAP/arm; ls -la $CAP"
docker logs dsv41-karmic-kraken 2>&1 | grep "mhc capture probe" | head -8
rm -rf "$out/captures" && mkdir -p "$out/captures" && cp cache/kkref/mhc-capture/mhc-*.pt "$out/captures/"
ls "$out/captures"
stop_all
log "replay"
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py $E/mhc_capture_replay.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host \
  -v $HOME/spark3-overlay/ref3m/vllm/models/deepseek_v4_1/b12x_layers.py:/opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -v $PWD/$out/captures:/cap:ro -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 -e B12X_AUTOTUNE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/mhc_capture_replay.py /cap \
  > "$out/replay.txt" 2>&1
log "replay exit $?"
grep -hE '"groups"|"agreement"|"timing_us"|"config"|"captures"|done|Error|Traceback' "$out/replay.txt" | cut -c1-300 | head -60
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
