#!/bin/bash
# usage: run43.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8, gemv-lookup-mhc-bi-fp8-rs and attn-exact3
#                    overlays on every node)
# The bench's eight concurrent copies of one prompt split with every fix, by how each stream was
# prefilled. detm-r5o-rs-trace, c8_trace.py (two rounds of eight identical ~70-token requests,
# 16 tokens); restores r5o, then analyze_trace3.py (every stream against stream 0) on every rank.
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
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
log "start c8 trace"
bin/spark3 --cluster-config $E/cluster-detm-r5o-rs-trace.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8" --rounds 2 --tokens 16 | tee "$out/c8-runs.jsonl"
log "c8 exit ${PIPESTATUS[0]}"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
for node in dgx1 dgx2 dgx3; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/c8:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t c8 --node $node --chain 12 2>&1 | grep -v Warn \
    > "$out/analysis3-c8-$node.jsonl"
done
log "done"
