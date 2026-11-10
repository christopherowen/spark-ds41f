#!/bin/bash
# usage: fit_tp3.sh <seed>   (CPU only; through run.sh)
set -euo pipefail
W=$HOME/spark-lab/agents/round3-957p5
seed=$1
exec "$W/scripts/run.sh" "fit-tp3-seed$seed" --rm --name "r3-957p5-fit$seed" --network none --cpus 4 \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$W/in:/w/in:ro" -v "$W/scripts:/w/scripts:ro" -v "$W/out:/w/out" -w /w \
  --entrypoint python3 spark-ds41f:candidate-50 /w/scripts/fit_tp3.py \
  --replay986 /w/in/986-replay_budget.py --served /w/in/986-served_targets.json --fit986 "/w/in/986-replay-seed$seed.json" \
  --tp3-c8 /w/in/served/tp3-c8.json /w/in/served/tp3-c8-again.json --tp3-c1 /w/in/served/tp3-c1.json \
  --seed "$seed" --out "/w/out/fit-tp3-seed$seed.json"
