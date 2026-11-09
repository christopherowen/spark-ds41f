#!/usr/bin/env bash
# usage: quiet2.sh <container-name> <command...>   (run inside flock /tmp/spark-lab-gpu.lock)
# quiet.sh plus: each foreign container seen during the run is inspected once for GPU
# device requests. Prints "foreign_containers_during_run=name:gpu|nogpu|gone,..." and
# "foreign_gpu_containers_during_run=..." (empty = no GPU job shared the run).
name=$1; shift
t0=$(date +%s)
while [ -n "$(docker ps -q)" ]; do sleep 5; done
echo "quiet_wait_s=$(( $(date +%s) - t0 ))"
log=$(mktemp)
(
  declare -A seen
  while true; do
    for n in $(docker ps --format "{{.Names}}"); do
      [ "$n" = "$name" ] && continue
      [ -n "${seen[$n]:-}" ] && continue
      dr=$(docker inspect --format "{{.HostConfig.DeviceRequests}}" "$n" 2>/dev/null) || dr=gone
      case "$dr" in gone) k=gone;; *gpu*) k=gpu;; *) k=nogpu;; esac
      seen[$n]=1; echo "$n:$k" >> "$log"
    done
    sleep 1
  done
) &
sampler=$!
"$@"
rc=$?
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
echo "foreign_containers_during_run=$(sort -u "$log" | paste -sd, -)"
echo "foreign_gpu_containers_during_run=$(grep -v -E ":nogpu$" "$log" | sort -u | paste -sd, -)"
rm -f "$log"
exit $rc
