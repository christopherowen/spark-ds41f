#!/usr/bin/env bash
# usage: quiet.sh <container-name> <command...>   (run inside flock /tmp/spark-lab-gpu.lock)
# Waits until no container is running on the host (an unlocked GPU job would share the
# GPU with ours), runs the command, and samples `docker ps` every 2 s while it runs.
# Prints "quiet_wait_s=..." and "foreign_containers_during_run=..." (empty = clean).
name=$1; shift
t0=$(date +%s)
while [ -n "$(docker ps -q)" ]; do sleep 5; done
echo "quiet_wait_s=$(( $(date +%s) - t0 ))"
log=$(mktemp)
( while true; do docker ps --format "{{.Names}}" >> "$log"; sleep 2; done ) &
sampler=$!
"$@"
rc=$?
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
echo "foreign_containers_during_run=$(sort -u "$log" | grep -vx "$name" | paste -sd, -)"
rm -f "$log"
exit $rc
