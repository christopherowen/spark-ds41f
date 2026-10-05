#!/bin/bash
# usage: reference.sh   (on dgx1, deployment checkout at this experiment's commit)
# Runs the reference bench on the running cluster-candidate-auto.json arm (the
# configuration promoted as r5k) into results/private/bench/r5k-reference, plus
# real-text prefill to 200K. Starts nothing; the arm must already be up.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-r5k
C=$E/cluster-candidate-auto.json
log() { echo "$(date -u +%FT%TZ) $*"; }
bin/spark --cluster-config $C doctor --live || true
bin/spark --cluster-config $C bench --allow-mismatch --compare none \
  --suites quality,decode,prefill,prefix,admission \
  --decode-cases prose,code,prose-nothink,code-nothink,json-nothink \
  --output results/private/bench/r5k-reference
log "bench exit $?"
bin/spark --cluster-config $C bench --allow-mismatch --compare none --suites prefill \
  --prefill-text source --prefill-sizes 4096,16384,32768,65536,131072,200000 --prefill-repeats 2 \
  --output results/private/bench/r5k-prefill-source
log "real-text prefill exit $?"
