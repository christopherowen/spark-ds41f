#!/usr/bin/env bash
set -euo pipefail
cd /home/swank/projects/spark-ds41f

rm -rf results/private/suites/1107-release-tp3/spark-ds41f-1107-again \
       results/private/suites/1107-release-tp3-single-stream/spark-ds41f-1107-again \
       results/private/suites/1107-release-tp3-sampled/spark-ds41f-1107-again

echo "=== [1/5] Boot 2 release-tp3 (spark-ds41f-1107-again) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp3-dev.json bench \
  --suite release-tp3 \
  --compare none \
  --corpus results/private/suites/1107-release-tp3/spark-ds41f-1107/model.json.gz \
  --output results/private/suites/1107-release-tp3/spark-ds41f-1107-again \
  --label spark-ds41f-1107-again

echo "=== [2/5] Boot 2 single-stream (spark-ds41f-1107-again) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp3-dev.json bench \
  --suite single-stream \
  --compare none \
  --output results/private/suites/1107-release-tp3-single-stream/spark-ds41f-1107-again \
  --label spark-ds41f-1107-again

echo "=== [3/5] Boot 2 sampled (spark-ds41f-1107-again) ==="
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp3-dev.json bench \
  --suite sampled \
  --compare none \
  --output results/private/suites/1107-release-tp3-sampled/spark-ds41f-1107-again \
  --label spark-ds41f-1107-again

echo "=== [4/5] Generating release-table.txt and suite manifest directories ==="
mkdir -p /home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3 \
         /home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3-single-stream \
         /home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3-sampled

for f in suite.json conditions.json grid.json c8.json c4.json poisson.json stall-3.json stall-6.json stall-7.json prefill.json; do
  cp -f "results/private/suites/1107-release-tp3/spark-ds41f-1107/$f" "/home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3/$f"
done
python3 scripts/servebench_suite_table.py 1107-release-tp3 spark-ds41f-1107 results/private/suites/candidate-14-release-tp3/spark-ds41f-1030 spark-ds41f-1107-again > /home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3/release-table.txt

for f in suite.json conditions.json c1.json; do
  cp -f "results/private/suites/1107-release-tp3-single-stream/spark-ds41f-1107/$f" "/home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3-single-stream/$f"
done
python3 scripts/servebench_suite_table.py 1107-release-tp3-single-stream spark-ds41f-1107 manifests/suites/2026-10-30-spark-ds41f-1030-tp3-single-stream spark-ds41f-1107-again > /home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3-single-stream/release-table.txt

for f in suite.json conditions.json c16.json c8.json c1.json; do
  cp -f "results/private/suites/1107-release-tp3-sampled/spark-ds41f-1107/$f" "/home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3-sampled/$f"
done
python3 scripts/servebench_suite_table.py 1107-release-tp3-sampled spark-ds41f-1107 manifests/suites/2026-10-30-spark-ds41f-1030-tp3-sampled spark-ds41f-1107-again > /home/swank/spark-lab/1107-tp3-triangle/suites/2026-11-07-spark-ds41f-1107-tp3-sampled/release-table.txt

echo "=== [5/5] Booting production 1107/tp3.json with telemetry ==="
python3 /home/swank/spark-lab/1107-tp3-triangle/telemetry/boot_with_telemetry.py \
  experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json \
  boot1-tp3-prod \
  /home/swank/spark-lab/1107-tp3-triangle/telemetry/boot1-tp3-prod.json

for n in dgx1 dgx2 dgx3; do
  ssh "$n" "sudo find /dev/shm -maxdepth 1 -type f -name \"psm_*\" -mmin +15 -delete || true"
done

echo "=== STEP 2B COMPLETE ==="
