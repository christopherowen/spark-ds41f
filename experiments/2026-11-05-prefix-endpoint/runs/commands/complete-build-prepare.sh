#!/bin/bash
set -euo pipefail
cd /home/chris/spark-lab/agents/prefix-endpoint/deployment
flock "$HOME/spark-lab/offline-cpu.lock" flock "$HOME/spark-cpu-work.lock" systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 taskset -c 0,1 nice -n 10 bin/spark --cluster-config experiments/2026-11-05-prefix-endpoint/tp3.json build prepare
