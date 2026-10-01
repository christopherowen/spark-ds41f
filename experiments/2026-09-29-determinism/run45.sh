#!/bin/bash
# usage: run45.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8, gemv-lookup-mhc-bi-fp8-rs2 and attn-exact4
#                    overlays on every node)
# The repaired harness, live: detm-r5o-rs2-exact4-trace (every fix, 0031 v2, attn-exact4 with wide
# logs of 4160 rows). c8_trace.py (one round: sequence-parallel WO and index records at their
# offsets on every rank), then scenario_trace.py (mixed prefill/decode, chunked 9000-token prompts,
# prefix-cache reuse, staggered identical requests; one repeat). The first dump writes each rank's
# plan inventory. Restores r5o, then analyze_trace4.py on every rank.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/harness
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
mkdir -p "$out"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 gemv-lookup-mhc-bi-fp8 gemv-lookup-mhc-bi-fp8-rs2 attn-exact4 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run45.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
log "start exact4 trace"
bin/spark3 --cluster-config $E/cluster-detm-r5o-rs2-exact4-trace.json cluster start --replace --apply | grep -v 'docker run'
for n in dgx1 dgx2 dgx3; do
  ssh -n $n "docker exec dsv41-karmic-kraken sh -c 'rm -f /cache/kkref/moe-checksums/inventory-rank*.json /cache/kkref/moe-checksums/plans-rank*.json'" || true
done
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8" --rounds 1 --tokens 16 | tee "$out/c8-runs.jsonl"
log "c8 exit ${PIPESTATUS[0]}"
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 | tee "$out/scenario-runs.jsonl"
log "scenarios exit ${PIPESTATUS[0]}"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
for node in dgx1 dgx2 dgx3; do
  for d in c8 scenarios; do
    docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace4.py:/a.py:ro -v $PWD/$out/$d:/t:ro \
      --entrypoint python3 $IMAGE /a.py /t --node $node --chain 12 2>&1 | grep -v Warn \
      > "$out/analysis4-$d-$node.jsonl"
  done
done
log "done"
