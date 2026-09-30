#!/bin/bash
# usage: run33.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with
#                    the det-variant, gemv-lookup, gemv-lookup-mhc and attn-exact overlays on every node)
# run32.sh's traces again, after both trace boots stopped below dgx1's memory guard:
# 1. Trace 3: detm-r5o-lookup-variant-exact (exact fingerprints; GEMV and MoE fixes), JSON and
#    prose, five mixes, two repeats.
# 2. Trace 4: detm-r5o-lookup-mhc-variant-exact (also the mHC fix), three repeats.
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
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find gemv-lookup-mhc attn-exact -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run33.txt"
done
for arm in "detm-r5o-lookup-variant-exact trace3 2" "detm-r5o-lookup-mhc-variant-exact trace4 3"; do
  set -- $arm
  stop_all
  log "start $1"
  bin/spark3 --cluster-config $E/cluster-$1.json cluster start --replace --apply | grep -v 'docker run'
  python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/$2" --repeats $3 --tokens 128 | tee "$out/$2-runs.jsonl"
  log "$2 exit ${PIPESTATUS[0]}"
  docker exec dsv41-karmic-kraken sh -c 'cat /cache/kkref/moe-checksums/plans-rank0.json' > "$out/plans-$2.json"
done
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
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
