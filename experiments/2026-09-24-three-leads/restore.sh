#!/bin/bash
# usage: restore.sh   (on dgx1) stop every arm and restart the promoted service
set -eu
cd ~/projects/spark3-vllm-ds41f
for config in experiments/2026-09-24-three-leads/cluster-*.json; do
  bin/spark --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live
