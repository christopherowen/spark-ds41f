#!/bin/bash
# Policy replay container (through run.sh: lock, hold-file and docker-ps checks).
# usage: replay.sh <name> <seed> <steps> [fit file under in/ or out/, default #986 r1 replay-seed<seed>.json]
set -euo pipefail
W=$HOME/spark-lab/agents/round3-957p5
name=$1; seed=$2; steps=$3; fit=${4:-/w/in/986-replay-seed$seed.json}
V=()
while read -r f; do V+=(-v "$W/ovl/c54/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < "$W/in/changed-c54.txt"
exec "$W/scripts/run.sh" "$name" --rm --name "r3-957p5-$name" --gpus all --network none --cpus 4 \
  -e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 -e PYTHONDONTWRITEBYTECODE=1 \
  "${V[@]}" -v "$W/in:/w/in:ro" -v "$W/scripts:/w/scripts:ro" -v "$W/out:/w/out" -w /w \
  --entrypoint python3 spark-ds41f:candidate-50 /w/scripts/replay_957p5.py \
  --c54 /w/in/av_c54.py --p5 /w/in/av_p5.py \
  --served /w/in/986-served_targets.json --fit "$fit" \
  --grid3 /w/in/grid/candidate-50-no1009-tp3-switched-screen.grid.json /w/in/grid/candidate-50-no1009-tp3-switched-screen-again.grid.json \
  --grid4 /w/in/grid/candidate-50-no1009-tp4-switched-screen.grid.json /w/in/grid/candidate-50-no1009-tp4-switched-screen-again.grid.json \
  --tp3-c8 /w/in/served/tp3-c8.json /w/in/served/tp3-c8-again.json --tp3-c1 /w/in/served/tp3-c1.json \
  --steps "$steps" --out "/w/out/$name.json"
