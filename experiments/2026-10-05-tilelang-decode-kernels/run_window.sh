#!/usr/bin/env bash
# One boot per arm: start, quality gate, single-stream decode profile, decode at
# one and eight streams (three samples), stop. Run from the Mac; the hold file
# on dgx1 must already be ours. Usage: run_window.sh OUT_DIR ARM...
set -uo pipefail
OUT=$1; shift
mkdir -p "$OUT"
R=/home/swank/projects/spark-ds41f
E=experiments/2026-10-05-tilelang-decode-kernels
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/runner.log"; }
remote() { ssh dgx1 "cd $R && $*"; }

for arm in "$@"; do
  if ssh dgx1 'ls ~/spark-request.json ~/spark3-request.json 2>/dev/null' | grep -q .; then
    log "request pending; stopping before $arm"; break
  fi
  log "$arm: start"
  remote "bin/spark --cluster-config $E/$arm.json cluster start --replace --apply" > "$OUT/start-$arm.txt" 2>&1
  status=$?
  tail -2 "$OUT/start-$arm.txt" | tee -a "$OUT/runner.log"
  if [ $status -ne 0 ]; then
    log "$arm: start failed ($status)"
    for h in dgx1 dgx2 dgx3 dgx4; do ssh $h 'docker logs dsv41-karmic-kraken 2>&1 | tail -60' > "$OUT/failed-$arm-$h.log" 2>&1; done
    remote "bin/spark --cluster-config $E/$arm.json cluster stop --apply --parallel" >> "$OUT/runner.log" 2>&1
    continue
  fi
  remote "bin/spark --cluster-config $E/$arm.json bench --url http://10.0.1.71:8000 --suites quality --output .work/decode-kernels/$arm-quality" > "$OUT/quality-$arm.txt" 2>&1
  grep -E "quality: LRU" "$OUT/quality-$arm.txt" | tee -a "$OUT/runner.log"
  remote "python3 experiments/2026-09-29-determinism/profile_decode.py http://10.0.1.71:8000 --tokens 160" >> "$OUT/runner.log" 2>&1
  remote "bin/spark --cluster-config $E/$arm.json bench --url http://10.0.1.71:8000 --suites decode --decode-cases prose,code --concurrency 1,8 --min-samples 3 --max-samples 3 --output .work/decode-kernels/$arm-decode" > "$OUT/decode-$arm.txt" 2>&1
  grep -E "^  (prose|code)-c|step time" -A0 "$OUT/decode-$arm.txt" | tee -a "$OUT/runner.log"
  scp -q "dgx1:$R/.work/decode-kernels/$arm-decode/bench.json" "$OUT/decode-$arm.json" 2>/dev/null
  remote "bin/spark --cluster-config $E/$arm.json cluster stop --apply --parallel" >> "$OUT/runner.log" 2>&1
  log "$arm: done"
done
log "window runs done"
