#!/bin/bash
# usage: run31.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with
#                    the det-variant, gemv-lookup and attn-probe overlays on every node)
# Boots detm-r5o-lookup-variant-probe and runs trace_mixes.py (JSON and prose targets, five
# background mixes, two repeats) with the query-projection recompute probe on, then
# analyze_probe.py and analyze_trace2.py --main. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
  experiments/2026-09-30-r5n/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && sha256sum attn-probe/*.py' | sed "s/^/$n /" >> "$out/overlay-bytes-run31.txt"
done
log "start probe"
bin/spark3 --cluster-config $E/cluster-detm-r5o-lookup-variant-probe.json cluster start --replace --apply \
  | grep -v 'docker run'
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/probe" --repeats 2 --tokens 128 | tee "$out/probe-runs.jsonl"
log "probe trace exit ${PIPESTATUS[0]}"
bin/spark3 --cluster-config $E/cluster-detm-r5o-lookup-variant-probe.json cluster stop --remove --apply >/dev/null 2>&1 || true
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
for script in analyze_probe.py; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/$script:/a.py:ro -v $PWD/$out/probe:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t 2>&1 | grep -v Warn | tee "$out/probe-analysis.txt"
done
for prompt in json prose; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace2.py:/a.py:ro -v $PWD/$out/probe:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t $prompt --main 2>&1 | grep -v Warn > "$out/probe-analysis-$prompt-main.jsonl"
done
log "done"
