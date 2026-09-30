#!/bin/bash
# usage: run41.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8 and attn-exact3 overlays on every node)
# The final validation's long target differed with its prefill step's row count (998 alone,
# 1009-1040 beside background decodes). Cluster stopped:
# 1. In parallel: prefill_replay.py (V4.1 GEMVs, block-FP8 query and Engram projections, mHC; a
#    998-row target alone and behind or ahead of k rows) on dgx2, moe_prefill_replay.py (the
#    deterministic MoE through serving's one plan, det-variant2) on dgx3.
# 2. detm-r5o-final-wide-trace (the final trace arm with attn-exact3, whose wide logs hold steps
#    of up to 1088 rows), trace_mixes.py for the long target, five mixes, two repeats, 4 tokens.
# Restores r5o, then distinct outputs and analyze_trace3.py on every rank.
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
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 gemv-lookup-mhc-bi-fp8 attn-exact3 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run41.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
replay() {  # node script output [docker args]
  ssh -n "$1" 'mkdir -p /tmp/lookup/cache'
  scp -q $E/$2 "$1":/tmp/lookup/
  ssh "$1" "IMAGE=$IMAGE SCRIPT=$2 EXTRA='$4' bash -s" > "$out/$3" 2>&1 <<'REMOTE'
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host $EXTRA \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/$SCRIPT
REMOTE
}
B=/opt/spark3/candidate/b12x
DET=""
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
  b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py; do
  DET="$DET -v /home/swank/spark3-overlay/det-variant2/$f:$B/$f:ro"
done
log "replays: prefill projections and mHC on dgx2, deterministic MoE on dgx3"
replay dgx2 prefill_replay.py prefill-replay.txt "" &
replay dgx3 moe_prefill_replay.py moe-prefill-replay.txt "$DET" &
wait
log "replays done"
grep -hvE "Warning|warn\(" "$out/prefill-replay.txt" "$out/moe-prefill-replay.txt" | grep -E "groups|^    |repeated|Error|Traceback|done" \
  | cut -c1-300 | head -80
log "start wide trace"
bin/spark3 --cluster-config $E/cluster-detm-r5o-final-wide-trace.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/wide" --repeats 2 --tokens 4 --prompts long \
  | tee "$out/wide-runs.jsonl"
log "wide exit ${PIPESTATUS[0]}"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
for node in dgx1 dgx2 dgx3; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/wide:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t long --node $node --chain 12 2>&1 | grep -v Warn \
    > "$out/analysis3-wide-long-$node.jsonl"
done
log "done"
