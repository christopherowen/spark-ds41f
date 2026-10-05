#!/bin/bash
# usage: run32.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with
#                    the det-variant, gemv-lookup, gemv-lookup-mhc and attn-exact overlays on every node)
# 1. On dgx3 (cluster stopped): mhc_serving_replay.py, the mHC operations under serving's
#    lookup and the smallest-capacity lookup.
# 2. Trace 3: detm-r5o-lookup-variant-exact (exact fingerprints; GEMV and MoE fixes), JSON and
#    prose, five mixes, two repeats.
# 3. Trace 4: detm-r5o-lookup-mhc-variant-exact (also the mHC fix), three repeats.
# Restores r5o, then analyze_trace2.py (all layers and --main) on both traces.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find gemv-lookup-mhc attn-exact -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run32.txt"
done
stop_all
ssh -n dgx3 'mkdir -p /tmp/lookup/cache'
scp -q $E/mhc_serving_replay.py dgx3:/tmp/lookup/
log "mhc replay on dgx3"
ssh dgx3 "IMAGE=$IMAGE bash -s" > "$out/mhc-serving-replay.txt" 2>&1 <<'REMOTE'
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/mhc_serving_replay.py
REMOTE
log "mhc replay done"
grep -E "^\[|^mHC|Error|Traceback" "$out/mhc-serving-replay.txt" | head -40
for arm in "detm-r5o-lookup-variant-exact trace3 2" "detm-r5o-lookup-mhc-variant-exact trace4 3"; do
  set -- $arm
  stop_all
  log "start $1"
  bin/spark --cluster-config $E/cluster-$1.json cluster start --replace --apply | grep -v 'docker run'
  python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/$2" --repeats $3 --tokens 128 | tee "$out/$2-runs.jsonl"
  log "$2 exit ${PIPESTATUS[0]}"
  docker exec dsv41-karmic-kraken sh -c 'cat /cache/kkref/moe-checksums/plans-rank0.json' > "$out/plans-$2.json"
done
stop_all
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
for t in trace3 trace4; do
  for prompt in json prose; do
    for mode in "" --main; do
      docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace2.py:/a.py:ro -v $PWD/$out/$t:/t:ro \
        --entrypoint python3 $IMAGE /a.py /t $prompt $mode 2>&1 | grep -v Warn \
        > "$out/analysis-$t-$prompt${mode:+-main}.jsonl"
    done
  done
done
log "done"
