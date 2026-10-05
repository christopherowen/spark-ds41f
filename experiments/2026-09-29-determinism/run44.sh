#!/bin/bash
# usage: run44.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8, gemv-lookup-mhc-bi-fp8-rs2 and attn-exact3
#                    overlays on every node)
# run43: between a small step and a sequence-parallel step the first difference was WO's reduction
# (one-shot all-reduce against 0031's reduce-scatter). detm-r5o-rs2-trace (0031 with the one-shot
# arithmetic): c8_trace.py (two rounds of eight identical requests, 16 tokens), then trace_mixes.py
# for the long target (five mixes, two repeats, 128 tokens). Restores r5o, then analyze_trace3.py
# on every rank for both.
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
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 gemv-lookup-mhc-bi-fp8 gemv-lookup-mhc-bi-fp8-rs2 attn-exact3 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run44.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
log "start rs2 trace"
bin/spark --cluster-config $E/cluster-detm-r5o-rs2-trace.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8-rs2" --rounds 2 --tokens 16 | tee "$out/c8-rs2-runs.jsonl"
log "c8 exit ${PIPESTATUS[0]}"
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/rs2" --repeats 2 --tokens 128 --prompts long \
  | tee "$out/rs2-runs.jsonl"
log "long exit ${PIPESTATUS[0]}"
stop_all
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
for node in dgx1 dgx2 dgx3; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/c8-rs2:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t c8 --node $node --chain 12 2>&1 | grep -v Warn \
    > "$out/analysis3-c8-rs2-$node.jsonl"
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/rs2:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t long --node $node --chain 12 2>&1 | grep -v Warn \
    > "$out/analysis3-rs2-long-$node.jsonl"
done
log "done"
