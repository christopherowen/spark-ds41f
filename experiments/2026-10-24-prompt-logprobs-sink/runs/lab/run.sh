#!/usr/bin/env bash
# The prompt-logprobs lab check (experiments/2026-10-24-prompt-logprobs-sink): wait until no
# window is held and no lab run is alive for two quiet minutes, sync the node checkouts to
# origin/main, run serving-check.json (the run restores production when it ends).
set -uo pipefail
repo=$HOME/projects/spark-ds41f
dir=$HOME/spark-lab/prompt-logprobs
spec=experiments/2026-10-24-prompt-logprobs-sink/serving-check.json
say() { echo "$(date -u +%FT%TZ) $*"; }
busy() { [ -e ~/spark-hold.json ] || pgrep -f "[s]cripts/lab.py (run|window|queue)" > /dev/null; }
cd "$repo" || exit 1
while :; do
  while busy; do sleep 30; done
  sleep 120
  busy && { say "busy again; waiting"; continue; }
  break
done
if [ -e ~/spark-request.json ]; then say "a request exists: $(cat ~/spark-request.json)"; exit 1; fi
git fetch -q origin || exit 1
bin/spark --cluster-config config/cluster.json cluster sync --revision origin/main --apply < /dev/null || exit 1
for i in $(seq 1 60); do
  python3 scripts/lab.py run "$spec" > "$dir/serving-check.out" 2>&1; code=$?
  tail -4 "$dir/serving-check.out"
  if grep -q "not opening" "$dir/serving-check.out"; then sleep 60; continue; fi
  say "serving-check.json exit $code"; break
done
bin/spark doctor --live 2>&1 | tail -1
say "PROMPT-LOGPROBS DONE"
