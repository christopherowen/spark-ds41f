#!/usr/bin/env bash
# Start the lab queue runner once production has been idle for MINUTES consecutive minutes
# (default 10), so a window never takes the cluster between an agent's requests.
#
# usage: scripts/lab_idle_start.sh [MINUTES] [LOG_NAME]
#
# A minute is idle when production's /metrics shows no running or waiting request and its
# finished-request counter has not moved since the previous minute. Then the runner starts
# (`queue run`, which itself refuses a busy cluster); the watcher exits once a window is held.
set -uo pipefail
need=${1:-10}
log_name=${2:-queue-lab}
repo=$(cd "$(dirname "$0")/.." && pwd)
idle=0
last=""
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
while true; do
  if [ -e "$HOME/spark-hold.json" ]; then say "a window is held; watcher done"; exit 0; fi
  metrics=$(curl -s -m 10 http://127.0.0.1:8000/metrics || true)
  sum() { awk -v name="$1" '$1 ~ "^"name"({|$)" {s += $2} END {printf "%d", s}' <<<"$metrics"; }
  running=$(sum vllm:num_requests_running)
  waiting=$(sum vllm:num_requests_waiting)
  finished=$(sum vllm:request_success_total)
  if [ -z "$metrics" ]; then
    idle=0
  elif [ "$running" = 0 ] && [ "$waiting" = 0 ] && [ "$finished" = "$last" ]; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  last=$finished
  say "running=$running waiting=$waiting finished=$finished idle_minutes=$idle"
  if [ "$idle" -ge "$need" ]; then
    say "idle for $need minutes; starting the queue runner"
    (cd "$repo" && setsid -f bash -c "python3 scripts/lab.py queue run --production-config config/cluster.json \
      >> results/private/lab/$log_name.log 2>&1" < /dev/null > /dev/null 2>&1)
    sleep 90
    if [ -e "$HOME/spark-hold.json" ]; then say "window open"; exit 0; fi
    idle=0
  fi
  sleep 60
done
