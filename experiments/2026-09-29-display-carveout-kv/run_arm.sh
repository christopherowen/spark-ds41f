#!/bin/bash
# usage: run_arm.sh ARM LABEL   (on dgx1, deployment checkout)
# ARM "control" benchmarks the running promoted service without restarting it.
# Any other ARM stops every service, starts cluster-ARM.json under the
# launcher's memory guards, and benchmarks it. Results go to
# results/private/bench/dcv-LABEL; the arm stays up.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-display-carveout-kv
arm=$1
label=$2
config=config/cluster.json
if [ "$arm" != control ]; then
  config=$E/cluster-$arm.json
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
  echo "$(date -u +%FT%TZ) start $arm as $label"
  bin/spark3 --cluster-config "$config" cluster start --replace --apply | grep -v 'docker run'
fi
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "echo \"\$(hostname): \$(awk '/MemAvailable/{printf \"%.2f GiB\", \$2/1048576}' /proc/meminfo) available;" \
    "docker logs dsv41-karmic-kraken 2>&1 | grep -E 'display carve-out|GPU KV cache size|Maximum concurrency' | tail -3\""
done
# --allow-mismatch: the arm's mounts and environment differ from the promoted config.
bin/spark3 --cluster-config "$config" bench --allow-mismatch --compare none \
  --suites quality,decode,prefill,admission --decode-cases prose,code,prose-nothink,code-nothink \
  --concurrency 1,8 --min-samples 3 --max-samples 3 --prefill-text source \
  --output "results/private/bench/dcv-$label"
