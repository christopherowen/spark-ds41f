#!/usr/bin/env bash
set -euo pipefail
cd /home/swank/projects/spark-ds41f

# Background heartbeat refresher every 5 minutes
(
  while true; do
    python3 -c "import json, datetime; p=\"/home/swank/spark-hold.json\"; d=json.load(open(p)); d[\"heartbeat\"]=datetime.datetime.now(datetime.timezone.utc).strftime(\"%Y-%m-%dT%H:%M:%SZ\"); json.dump(d, open(p,\"w\"), indent=2)" || true
    sleep 300
  done
) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

rm -rf results/private/suites/1107-release/spark-ds41f-1107 \
       results/private/suites/1107-release-single-stream/spark-ds41f-1107 \
       results/private/suites/1107-release-sampled/spark-ds41f-1107 \
       results/private/suites/1107-release/spark-ds41f-1107-again \
       results/private/suites/1107-release-single-stream/spark-ds41f-1107-again \
       results/private/suites/1107-release-sampled/spark-ds41f-1107-again

echo "=== [1/9] Boot 1 release (spark-ds41f-1107) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json bench \
  --suite release \
  --compare none \
  --corpus results/private/suites/candidate-12-release-tp4/spark-ds41f-1030/model.json.gz \
  --output results/private/suites/1107-release/spark-ds41f-1107 \
  --label spark-ds41f-1107

echo "=== [2/9] Boot 1 single-stream (spark-ds41f-1107) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json bench \
  --suite single-stream \
  --compare none \
  --output results/private/suites/1107-release-single-stream/spark-ds41f-1107 \
  --label spark-ds41f-1107

echo "=== [3/9] Boot 1 sampled (spark-ds41f-1107) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json bench \
  --suite sampled \
  --compare none \
  --output results/private/suites/1107-release-sampled/spark-ds41f-1107 \
  --label spark-ds41f-1107

echo "=== [4/9] Dev-Profile Diagnostics on 1107/tp4-dev.json ==="
python3 experiments/2026-10-25-final-names/loader/loader_report.py http://10.0.1.71:8000 --label 1107-tp4-dev --nodes dgx1,dgx2,dgx3,dgx4 > /home/swank/spark-lab/1107-tp4-ring/gate/loader_report.json
python3 experiments/2026-10-23-next-release/acceptance/copy_probe.py http://10.0.1.71:8000 --prompts 12 > /home/swank/spark-lab/1107-tp4-ring/gate/copy_probe.json
python3 experiments/2026-10-26-spark-ds41f-candidate/ced_cross_shape.py http://10.0.1.71:8000 > /home/swank/spark-lab/1107-tp4-ring/gate/ced_cross_shape.json
python3 experiments/2026-10-25-kv-zeroing/kv_counters.py http://10.0.1.71:8000 --nodes dgx1,dgx2,dgx3,dgx4 --container spark-ds41f > /home/swank/spark-lab/1107-tp4-ring/gate/kv_counters.json

echo "=== [5/9] Boot 2 (1107-again) restart with telemetry ==="
python3 /home/swank/spark-lab/1107-tp4-ring/telemetry/boot_with_telemetry.py \
  experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json \
  boot2-tp4-dev \
  /home/swank/spark-lab/1107-tp4-ring/telemetry/boot2-tp4-dev.json

for n in dgx1 dgx2 dgx3 dgx4; do
  ssh "$n" "sudo find /dev/shm -maxdepth 1 -type f -name \"psm_*\" -mmin +15 -delete || true"
done

echo "=== [6/9] Boot 2 release (spark-ds41f-1107-again) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json bench \
  --suite release \
  --compare none \
  --corpus results/private/suites/1107-release/spark-ds41f-1107/model.json.gz \
  --output results/private/suites/1107-release/spark-ds41f-1107-again \
  --label spark-ds41f-1107-again

echo "=== [7/9] Boot 2 single-stream (spark-ds41f-1107-again) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json bench \
  --suite single-stream \
  --compare none \
  --output results/private/suites/1107-release-single-stream/spark-ds41f-1107-again \
  --label spark-ds41f-1107-again

echo "=== [8/9] Boot 2 sampled (spark-ds41f-1107-again) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4-dev.json bench \
  --suite sampled \
  --compare none \
  --output results/private/suites/1107-release-sampled/spark-ds41f-1107-again \
  --label spark-ds41f-1107-again

echo "=== [9/9] Generating release-table.txt and suite manifest directories ==="
mkdir -p /home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4 \
         /home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4-single-stream \
         /home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4-sampled

for f in suite.json conditions.json grid.json c16.json c8.json poisson.json stall-3.json stall-11.json stall-15.json prefill.json; do
  cp -f "results/private/suites/1107-release/spark-ds41f-1107/$f" "/home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4/$f"
done
python3 scripts/servebench_suite_table.py 1107-release spark-ds41f-1107 results/private/suites/candidate-12-release-tp4/spark-ds41f-1030 spark-ds41f-1107-again > /home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4/release-table.txt

for f in suite.json conditions.json c1.json; do
  cp -f "results/private/suites/1107-release-single-stream/spark-ds41f-1107/$f" "/home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4-single-stream/$f"
done
python3 scripts/servebench_suite_table.py 1107-release-single-stream spark-ds41f-1107 manifests/suites/2026-10-30-spark-ds41f-1030-tp4-single-stream spark-ds41f-1107-again > /home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4-single-stream/release-table.txt

for f in suite.json conditions.json c16.json c8.json c1.json; do
  cp -f "results/private/suites/1107-release-sampled/spark-ds41f-1107/$f" "/home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4-sampled/$f"
done
python3 scripts/servebench_suite_table.py 1107-release-sampled spark-ds41f-1107 manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled spark-ds41f-1107-again > /home/swank/spark-lab/1107-tp4-ring/suites/2026-11-07-spark-ds41f-1107-tp4-sampled/release-table.txt

echo "=== STEP 2 COMPLETE ==="
