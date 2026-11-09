#!/usr/bin/env bash
# usage: sequence.sh <label:arm:topology:port> ...
# Runs run4.sh for each spec back to back (same session), stopping at the first run with a
# non-zero rank exit code.
set -u
W=$HOME/spark-lab/agents/candidate-49-rework-979-r1
for spec in "$@"; do
  IFS=: read -r label arm topo port <<< "$spec"
  echo "=== $(date -u +%FT%TZ) $label ($arm, $topo, port $port)"
  "$W/run4.sh" "$label" "$arm" "$topo" "$port" > "$W/out/driver-$label.log" 2>&1
  tail -n 20 "$W/out/driver-$label.log"
  rcline=$(grep '^rc:' "$W/out/driver-$label.log" || echo "rc: missing")
  if [ "$rcline" != "rc: 0 0 0 0" ]; then echo "=== stopping: $label $rcline"; exit 1; fi
done
echo "=== $(date -u +%FT%TZ) sequence done"
