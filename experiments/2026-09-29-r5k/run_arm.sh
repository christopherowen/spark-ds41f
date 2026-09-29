#!/bin/bash
# usage: run_arm.sh ARM LABEL [bench options...]   (on dgx1, deployment checkout)
# Stops every service, starts cluster-ARM.json under the launcher's memory
# guards, and benchmarks it into results/private/bench/r5k-LABEL. The arm
# stays up.
set -euo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-r5k
arm=$1
label=$2
shift 2
config=$E/cluster-$arm.json
for c in config/cluster.json $E/cluster-*.json experiments/2026-09-29-display-carveout-kv/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
echo "$(date -u +%FT%TZ) start $arm as $label"
bin/spark3 --cluster-config "$config" cluster start --replace --apply | grep -v 'docker run'
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "echo \"\$(hostname): \$(sudo -n cat /sys/kernel/debug/dri/0/clients | tail -n +2 | wc -l) DRM clients\";" \
    "docker logs dsv41-karmic-kraken 2>&1 | grep -E 'Display carve-out holds|GPU KV cache size' | cut -c1-220 | tail -2"
done
# --allow-mismatch: the arm differs from the promoted config.
bin/spark3 --cluster-config "$config" bench --allow-mismatch --compare none \
  --suites quality,decode,prefill,admission --decode-cases prose,code,prose-nothink,code-nothink \
  --concurrency 1,8 --min-samples 3 --max-samples 3 --prefill-text source \
  --output "results/private/bench/r5k-$label" "$@"
