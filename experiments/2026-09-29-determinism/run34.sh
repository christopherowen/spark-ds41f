#!/bin/bash
# usage: run34.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant, gemv-lookup-mhc, gemv-lookup-mhc-bi and attn-exact overlays on every node)
# 1. On dgx3 (cluster stopped): gemv_backend_replay.py (BF16-output GEMVs on the default and the
#    SIMT backend, and the index-key RMSNorm).
# 2. Trace 4: detm-r5o-lookup-mhc-variant-exact (exact fingerprints; GEMV, MoE and mHC fixes),
#    JSON and prose, five mixes, two repeats.
# 3. Trace 5: detm-r5o-lookup-mhc-bi-variant-exact (also SIMT for the torch-backend GEMVs),
#    three repeats.
# Restores r5o, then analyze_trace2.py (all layers, --main, --chain 12) on both traces.
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
  ssh -n "$n" 'cd spark3-overlay && find gemv-lookup-mhc gemv-lookup-mhc-bi attn-exact -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run34.txt"
done
stop_all
ssh -n dgx3 'mkdir -p /tmp/lookup/cache'
scp -q $E/gemv_backend_replay.py dgx3:/tmp/lookup/
log "backend replay on dgx3"
ssh dgx3 "IMAGE=$IMAGE bash -s" > "$out/gemv-backend-replay.txt" 2>&1 <<'REMOTE'
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/gemv_backend_replay.py
REMOTE
log "backend replay done"
grep -vE "Warning|warn\(" "$out/gemv-backend-replay.txt" | grep -E "groups|us;|: [0-9]|Error|Traceback" | head -30
for arm in "detm-r5o-lookup-mhc-variant-exact trace4 2" "detm-r5o-lookup-mhc-bi-variant-exact trace5 3"; do
  set -- $arm
  stop_all
  log "start $1"
  bin/spark3 --cluster-config $E/cluster-$1.json cluster start --replace --apply | grep -v 'docker run'
  for n in dgx1 dgx2 dgx3; do  # fresh plan dumps for this boot
    ssh -n $n "docker exec dsv41-karmic-kraken sh -c 'rm -f /cache/kkref/moe-checksums/plans-rank*.json'" || true
  done
  python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/$2" --repeats $3 --tokens 128 | tee "$out/$2-runs.jsonl"
  log "$2 exit ${PIPESTATUS[0]}"
  docker exec dsv41-karmic-kraken sh -c 'cat /cache/kkref/moe-checksums/plans-rank0.json' > "$out/plans-$2.json"
done
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
for t in trace4 trace5; do
  for prompt in json prose; do
    for mode in "" --main; do
      docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace2.py:/a.py:ro -v $PWD/$out/$t:/t:ro \
        --entrypoint python3 $IMAGE /a.py /t $prompt $mode --chain 12 2>&1 | grep -v Warn \
        > "$out/analysis-$t-$prompt${mode:+-main}.jsonl"
    done
  done
done
log "done"
