#!/bin/bash
# usage: run_arm.sh ARM LABEL   (on dgx1, from the deployment checkout)
# Starts cluster-ARM.json under the launcher's memory guards, runs the quality
# gate, single-stream decode and real-text prefill into
# results/private/bench/sp-LABEL, then the long-prompt checks (sp_check.py).
set -eu
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-27-prefill-sp
arm=$1
label=$2
for config in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
echo "$(date -u +%FT%TZ) start $arm as $label (dgx1 MemAvailable ${avail} GiB)"
bin/spark --cluster-config "$E/cluster-$arm.json" cluster start --replace --apply | grep -v 'docker run'
bin/spark --cluster-config "$E/cluster-$arm.json" bench --allow-mismatch --compare none \
  --suites quality,decode,prefill --decode-cases prose-nothink,code-nothink,explain --concurrency 1 \
  --min-samples 3 --max-samples 3 --prefill-text source --prefill-sizes 4096,16384,32768,65536 \
  --prefill-repeats 2 --output "results/private/bench/sp-$label" || true
python3 $E/sp_check.py http://10.0.1.71:8000 "results/private/bench/sp-$label/sp_check.json"
