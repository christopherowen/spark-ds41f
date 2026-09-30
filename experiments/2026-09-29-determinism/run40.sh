#!/bin/bash
# usage: run40.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8 and attn-exact2 overlays on every node)
# Final validation: detm-r5o-final-trace (deterministic MoE with every fix, batch-invariant mode,
# exact fingerprints with row positions and tokens), trace_mixes.py for the JSON, prose and long
# (about a thousand prompt tokens) targets, five mixes, three repeats. Restores r5o, then counts
# distinct outputs and runs analyze_trace3.py on every rank.
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
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 gemv-lookup-mhc-bi-fp8 attn-exact2 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run40.txt"
done
stop_all
log "start final trace"
bin/spark3 --cluster-config $E/cluster-detm-r5o-final-trace.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/final" --repeats 3 --tokens 128 --prompts json,prose,long \
  | tee "$out/final-runs.jsonl"
log "final exit ${PIPESTATUS[0]}"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
for prompt in json prose long; do
  for node in dgx1 dgx2 dgx3; do
    docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/final:/t:ro \
      --entrypoint python3 $IMAGE /a.py /t $prompt --node $node --chain 10 2>&1 | grep -v Warn \
      > "$out/analysis3-final-$prompt-$node.jsonl"
  done
done
log "done"
