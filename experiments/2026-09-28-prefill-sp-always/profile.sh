#!/bin/bash
# usage: profile.sh   (on dgx1, deployment checkout at this experiment's commit)
# One boot each of current-profile and always2-profile (or \$ARMS); capture_tiny.py records
# six ~73-token prefills per arm. Traces land in cache/kkref/profiles/spa-ARM/.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-prefill-sp-always
log() { echo "$(date -u +%FT%TZ) $*"; }
for arm in ${ARMS:-current-profile always2-profile}; do
  for config in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
  for _ in $(seq 1 60); do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 100 ] && break
    sleep 5
  done
  log "start $arm (dgx1 MemAvailable ${avail} GiB)"
  bin/spark3 --cluster-config "$E/cluster-$arm.json" cluster start --replace --apply | grep -v 'docker run'
  python3 $E/capture_tiny.py http://10.0.1.71:8000
  log "arm $arm exit $?"
  sleep 30  # let every rank finish writing its trace
done
