#!/bin/bash
# usage: run35.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant, gemv-lookup-mhc-bi and attn-probe2 overlays on every node)
# 1. Cluster stopped, in parallel: mhc_serving_replay.py with small targets (6 and 3 rows) on
#    dgx3, engram_wkv_replay.py on dgx2.
# 2. detm-r5o-lookup-mhc-bi-variant-probe (trace 5's arm with the attention and query-projection
#    recompute probes), trace_mixes.py JSON and prose, five mixes, one repeat.
# Restores r5o, then analyze_probe.py and analyze_trace2.py --main --chain 12.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && sha256sum attn-probe2/*.py' | sed "s/^/$n /" >> "$out/overlay-bytes-run35.txt"
done
stop_all
replay() {  # node script output [env]
  ssh -n "$1" 'mkdir -p /tmp/lookup/cache'
  scp -q $E/$2 "$1":/tmp/lookup/
  ssh "$1" "IMAGE=$IMAGE SCRIPT=$2 EXTRA='$4' bash -s" > "$out/$3" 2>&1 <<'REMOTE'
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host $EXTRA \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/$SCRIPT
REMOTE
}
log "replays: mhc small targets on dgx3, engram wkv on dgx2"
replay dgx3 mhc_serving_replay.py mhc-serving-replay-small.txt "-e MHC_TARGETS=6,3" &
replay dgx2 engram_wkv_replay.py engram-wkv-replay.txt "" &
wait
log "replays done"
grep -hE "^\[|^layer|Error|Traceback" "$out/mhc-serving-replay-small.txt" "$out/engram-wkv-replay.txt" | cut -c1-300 | head -40
log "start probe"
bin/spark3 --cluster-config $E/cluster-detm-r5o-lookup-mhc-bi-variant-probe.json cluster start --replace --apply \
  | grep -v 'docker run'
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/probe2" --repeats 1 --tokens 128 | tee "$out/probe2-runs.jsonl"
log "probe2 exit ${PIPESTATUS[0]}"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_probe.py:/a.py:ro -v $PWD/$out/probe2:/t:ro \
  --entrypoint python3 $IMAGE /a.py /t 2>&1 | grep -v Warn | tee "$out/probe2-analysis.txt"
for prompt in json prose; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace2.py:/a.py:ro -v $PWD/$out/probe2:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t $prompt --main --chain 12 2>&1 | grep -v Warn \
    > "$out/analysis-probe2-$prompt-main.jsonl"
done
log "done"
