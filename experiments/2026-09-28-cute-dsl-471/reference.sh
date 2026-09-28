#!/bin/bash
# usage: reference.sh   (on dgx1, deployment checkout at this experiment's commit)
# Starts cluster-final.json and runs the reference bench into
# results/private/bench/r5i-reference plus real-text prefill. The arm stays up.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-cute-dsl-471
log() { echo "$(date -u +%FT%TZ) $*"; }
for config in config/cluster.json $E/cluster-*.json; do
  bin/spark3 --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark3 --cluster-config $E/cluster-final.json cluster start --replace --apply | grep -v "docker run"
bin/spark3 --cluster-config $E/cluster-final.json doctor --live || true
bin/spark3 --cluster-config $E/cluster-final.json bench --allow-mismatch --compare none \
  --suites quality,decode,prefill,prefix,admission \
  --decode-cases prose,code,prose-nothink,code-nothink,json-nothink \
  --output results/private/bench/r5i-reference
log "bench exit $?"
bin/spark3 --cluster-config $E/cluster-final.json bench --allow-mismatch --compare none --suites prefill \
  --prefill-text source --prefill-sizes 4096,16384,32768,65536 --prefill-repeats 2 \
  --output results/private/bench/r5i-prefill-source
log "real-text prefill exit $?"
