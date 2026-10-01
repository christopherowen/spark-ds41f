#!/bin/bash
# usage: run51.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref1 and attn-exact6 overlays on every node)
# run50: under the reference configuration the 8093-token chunked prompt's first logprob differed
# (alone against beside other requests) with every recorded row equal through layer 39.
# detm-r5o-ref-trace6 records the LM head's input rows and logits: scenario_trace.py chunked and
# mixed once. analyze_trace4.py while the cluster is stopped. Then the reference configuration's
# cost by component: detm-r5o-ref-no{moe,attn,head,mhc,gemv}-pin, each leaving one change out, decode
# (prose and JSON, one stream, three samples) and cold prefill of real text (16384 and 65536 tokens,
# three repeats), one boot each, pinned cost table. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/ref6
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
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref1 ref1-var attn-exact6 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run51.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
log "start ref trace6"
bin/spark3 --cluster-config $E/cluster-detm-r5o-ref-trace6.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --scenarios chunked,mixed --repeats 1 \
  | tee "$out/scenario-runs.jsonl"
log "scenarios exit ${PIPESTATUS[0]}"
stop_all
for node in dgx1 dgx2 dgx3; do
  docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace4.py:/a.py:ro \
    -v $PWD/$out/scenarios:/t:ro --entrypoint python3 $IMAGE /a.py /t --node $node --chain 12 --all-pairs 2>&1 \
    | grep -v Warn > "$out/analysis4-scenarios-$node.jsonl"
done
log "analysed"
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
for v in nomoe noattn nohead nomhc nogemv; do
  stop_all
  log "start ref-$v"
  bin/spark3 --cluster-config $E/cluster-detm-r5o-ref-$v-pin.json cluster start --replace --apply | grep -v 'docker run'
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 >> "$out/pin.log"
  bin/spark3 --cluster-config $E/cluster-detm-r5o-ref-$v-pin.json bench --allow-mismatch --compare none \
    --suites decode,prefill --decode-cases prose,json-nothink --concurrency 1 --min-samples 3 --max-samples 3 \
    --prefill-text source --prefill-sizes 16384,65536 --prefill-repeats 3 --output "results/private/bench/ref-$v"
  log "bench ref-$v exit $?"
done
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
