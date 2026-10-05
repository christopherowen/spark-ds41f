#!/bin/bash
# usage: run42.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant, det-variant2, gemv-lookup-mhc-bi-fp8, gemv-lookup-mhc-bi-fp8-rs and
#                    attn-exact3 overlays on every node)
# 1. Validation: detm-r5o-rs-trace (every fix, vllm-0031's rank-order reduce-scatter, wide logs),
#    trace_mixes.py for the JSON, prose and long targets, five mixes, three repeats, 128 tokens.
# 2. Performance, no debug overlay, one pinned cost table, one boot per arm: decode (prose and
#    JSON answers, one and eight streams, three samples) and cold prefill of real text (1024, 4096,
#    16384 and 65536 tokens, three repeats) for r5o-pin, detm-r5o-bi-pin and detm-r5o-bi-rs-pin.
# Restores r5o, then distinct outputs and analyze_trace3.py on every rank.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
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
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant det-variant2 gemv-lookup-mhc-bi-fp8 gemv-lookup-mhc-bi-fp8-rs attn-exact3 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run42.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
start $E/cluster-detm-r5o-rs-trace.json
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/rs" --repeats 3 --tokens 128 --prompts json,prose,long \
  | tee "$out/rs-runs.jsonl"
log "rs trace exit ${PIPESTATUS[0]}"
measure() {  # arm label
  start $E/cluster-$1.json
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
  bin/spark --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 3 --max-samples 3 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/rs-$2"
  log "bench $2 exit $?"
}
measure r5o-pin r5o
measure detm-r5o-bi-pin detm-bi
measure detm-r5o-bi-rs-pin detm-bi-rs
stop_all
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
python3 - "$out/rs" <<'PY'
import glob, json, os, sys
for prompt in ("json", "prose", "long"):
    outs = {}
    for f in sorted(glob.glob(f"{sys.argv[1]}/{prompt}-m*-r*/target.json")):
        outs.setdefault(tuple(json.load(open(f))["tokens_ids"]), []).append(os.path.basename(os.path.dirname(f)))
    print(prompt, sum(map(len, outs.values())), "runs,", len(outs), "distinct outputs:",
          " | ".join(" ".join(n.replace(prompt + "-", "") for n in names) for names in outs.values()))
PY
for prompt in json prose long; do
  for node in dgx1 dgx2 dgx3; do
    docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/rs:/t:ro \
      --entrypoint python3 $IMAGE /a.py /t $prompt --node $node --chain 10 2>&1 | grep -v Warn \
      > "$out/analysis3-rs-$prompt-$node.jsonl"
  done
done
log "done"
