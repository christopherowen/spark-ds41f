#!/bin/bash
# The sp8 attribution queue: one boot per flock hold of /tmp/spark-lab-gpu.lock (run2.sh),
# ABBA over the arms, TP3 rank shapes, all-reduce stand-in 30 us, carve-out on (TP3 recipe).
# A smoke boot of candidate-54 first: the queue stops if it fails.
set -u
W=/home/chris/spark-lab/agents/tp3-grid-attribution
R=$W/tools/run2.sh
log=$W/out/runs/queue.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $log; }
say "queue start"
$R c54 1 30 || { say "smoke boot failed: stop"; exit 1; }
say "smoke ok"
arms=(noAll no961 no998 no960 no963-no1030)
for a in "${arms[@]}"; do $R $a 1 30 && say "$a 1 ok" || say "$a 1 FAILED"; done
for ((i=${#arms[@]}-1; i>=0; i--)); do a=${arms[$i]}; $R $a 2 30 && say "$a 2 ok" || say "$a 2 FAILED"; done
$R c54 2 30 && say "c54 2 ok" || say "c54 2 FAILED"
say "queue done"
