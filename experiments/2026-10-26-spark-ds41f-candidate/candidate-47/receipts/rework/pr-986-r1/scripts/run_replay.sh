#!/bin/bash
# Replay of the DSpark draft budget (CPU work in the 1107 image; no GPU kernels).
# Waits for the shared GPU lock, then requires an empty `docker ps` and no hold file.
set -euo pipefail
D=$HOME/spark-lab/agents/candidate-47-rework-986
exec 9>/tmp/spark-lab-gpu.lock
flock 9
if [ -e "$HOME/spark-hold.json" ]; then echo "hold file present; not running" >&2; exit 3; fi
if [ -n "$(docker ps -q)" ]; then echo "docker ps not empty; not running" >&2; docker ps >&2; exit 4; fi
date -u +"start %FT%TZ"
docker run --rm --gpus all --network none --cpus 6 --entrypoint python3 \
  -v "$D":/w spark-ds41f:1107 /w/in/replay_budget.py \
  --grid /w/in/grid/candidate-47.grid.json /w/in/grid/candidate-47-again.grid.json \
         /w/in/grid/candidate-47-no986.grid.json /w/in/grid/candidate-47-no986-again.grid.json \
  --served /w/in/served_targets.json --regress /w/in/out-batch-regress.json \
  --modules /w/in/av_1107.py /w/in/av_986.py --steps "${STEPS:-20000}" --seed "${SEED:-0}" --out "/w/out/replay-seed${SEED:-0}.json"
date -u +"end %FT%TZ"
