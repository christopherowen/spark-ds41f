#!/usr/bin/env bash
set -euo pipefail
cd /home/swank/projects/spark-ds41f

echo "=== [1] Boot 1 sampled suite ==="
rm -rf results/private/suites/1107-release-tp3-sampled/spark-ds41f-1107
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp3-dev.json bench \
  --suite sampled \
  --compare none \
  --output results/private/suites/1107-release-tp3-sampled/spark-ds41f-1107 \
  --label spark-ds41f-1107

echo "=== [2] Dev-Profile Diagnostics on 1107/tp3-dev.json ==="
python3 experiments/2026-10-25-final-names/loader/loader_report.py http://10.0.1.71:8000 --label 1107-tp3-dev --nodes dgx1,dgx2,dgx3 > /home/swank/spark-lab/1107-tp3-triangle/gate/loader_report.json
python3 experiments/2026-10-23-next-release/acceptance/copy_probe.py http://10.0.1.71:8000 --prompts 12 > /home/swank/spark-lab/1107-tp3-triangle/gate/copy_probe.json
python3 experiments/2026-10-26-spark-ds41f-candidate/ced_cross_shape.py http://10.0.1.71:8000 > /home/swank/spark-lab/1107-tp3-triangle/gate/ced_cross_shape.json
python3 experiments/2026-10-25-kv-zeroing/kv_counters.py http://10.0.1.71:8000 --nodes dgx1,dgx2,dgx3 --container spark-ds41f > /home/swank/spark-lab/1107-tp3-triangle/gate/kv_counters.json

echo "=== [3] Boot 2 (1107-again) restart with telemetry ==="
python3 /home/swank/spark-lab/1107-tp3-triangle/telemetry/boot_with_telemetry.py \
  experiments/2026-10-26-spark-ds41f-candidate/1107/tp3-dev.json \
  boot2-tp3-dev \
  /home/swank/spark-lab/1107-tp3-triangle/telemetry/boot2-tp3-dev.json

# Clean any orphaned /dev/shm/psm_* from Boot 1 across dgx1..dgx3
for n in dgx1 dgx2 dgx3; do
  ssh "$n" "sudo find /dev/shm -maxdepth 1 -type f -name \"psm_*\" -mmin +15 -delete || true"
done

echo "=== STEP 2A COMPLETE ==="
