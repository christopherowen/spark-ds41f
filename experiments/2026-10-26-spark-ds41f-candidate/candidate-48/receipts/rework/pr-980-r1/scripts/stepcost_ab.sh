#!/bin/bash
# stepcost_ab.sh <boot> [extra args]: the in-process A/B/C step-cost bench, the three arms` copies
# of the changed modules loaded side by side from ov/<arm> on spark-ds41f:candidate-48.
W=$HOME/spark-lab/agents/candidate-48-rework-980
IMG=sha256:d6ffd842978849651e756ebd39e9bf2da7ddb03e746113013533029878485a24
boot=$1; shift
exec flock /tmp/spark-lab-gpu.lock timeout 2400 docker run --rm --name c48rw980-ab-$boot --gpus all --ipc=host --network=host \
  -v $W/ov:/ab:ro -v $W/stepcost_ab.py:/tmp/stepcost_ab.py:ro -v $W/out:/out \
  --entrypoint python3 $IMG /tmp/stepcost_ab.py --root /ab --boot $boot --out /out/stepcost-ab-$boot.json "$@"
