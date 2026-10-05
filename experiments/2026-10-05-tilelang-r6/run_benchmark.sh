#!/usr/bin/env bash
# r6 TP4 acceptance benchmark: one boot of tp4-profile.json; quality, the
# single-stream decode profile, decode on prose and code at 1-16 streams
# (three samples) and real-text prefill at 32K, 256K, 500K and 1M (two
# repeats). Leaves the container running. Run from the Mac with the hold file
# on dgx1 ours. Usage: run_benchmark.sh OUT_DIR
set -uo pipefail
OUT=$1
mkdir -p "$OUT"
R=/home/swank/projects/spark-ds41f
E=experiments/2026-10-05-tilelang-r6
CFG=$E/tp4-profile.json
URL=http://10.0.1.71:8000
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/runner.log"; }
remote() { ssh dgx1 "cd $R && $*"; }

if ssh dgx1 'ls ~/spark-request.json ~/spark3-request.json 2>/dev/null' | grep -q .; then
  log "request pending; not starting"; exit 1
fi
log "start $CFG"
remote "bin/spark --cluster-config $CFG cluster start --replace --apply" > "$OUT/start.txt" 2>&1
status=$?
tail -2 "$OUT/start.txt" | tee -a "$OUT/runner.log"
if [ $status -ne 0 ]; then
  log "start failed ($status)"
  for h in dgx1 dgx2 dgx3 dgx4; do ssh $h 'docker logs dsv41-karmic-kraken 2>&1 | tail -80' > "$OUT/failed-$h.log" 2>&1; done
  exit $status
fi
remote "bin/spark --cluster-config $CFG bench --url $URL --suites quality --output .work/r6/quality" > "$OUT/quality.txt" 2>&1
grep -E "quality: LRU" "$OUT/quality.txt" | tee -a "$OUT/runner.log"
remote "python3 experiments/2026-09-29-determinism/profile_decode.py $URL --tokens 160" >> "$OUT/runner.log" 2>&1
log "bench decode + prefill"
remote "bin/spark --cluster-config $CFG bench --url $URL --suites decode,prefill --decode-cases prose,code --concurrency 1,2,4,8,16 --min-samples 3 --max-samples 3 --prefill-text source --prefill-sizes 32768,262144,500000,1048576 --prefill-repeats 2 --output .work/r6/bench" > "$OUT/bench.txt" 2>&1
tail -40 "$OUT/bench.txt" | tee -a "$OUT/runner.log"
scp -q "dgx1:$R/.work/r6/bench/bench.json" "$OUT/bench.json" 2>/dev/null
log "benchmark done; container left running"
