#!/usr/bin/env bash
# The memory window's dgx1 watcher (from experiments/2026-10-21-startup-memory/watch.sh, for
# the four-node ring):
#  1. when a queue runner starts <UTC>-next-memory-tp4.json (queued after this watcher
#     started), arm the read-only samplers on dgx1-4;
#  2. when the queue idles after that job, ask for the window back (the runner restores
#     production);
#  3. once the hold is gone and the restore boot has settled, stop the samplers and copy
#     every node's samples to dgx1.
# Reads only queue-log bytes written after it started. Run it from ~/spark-lab/next-memory,
# where sampler.py was copied on every node.
set -uo pipefail
spec=next-memory-tp4
dir=$HOME/spark-lab/next-memory
logs=$HOME/projects/spark-ds41f/results/private/lab
me="next-memory (soak and start-up memory, dgx1 watcher): queue idle after my spec, releasing"
settle=${SETTLE_SECONDS:-240}
since=$(date -u +%Y%m%dT%H%M%S)
say() { echo "$(date -u +%FT%TZ) $*"; }
declare -A off
for f in "$logs"/*.log; do off[$f]=$(stat -c %s "$f"); done
new_lines() {  # every queue-log line appended since the watcher started (a new log from its start)
  for f in "$logs"/*.log; do
    tail -c +$(( ${off[$f]:-0} + 1 )) "$f" | sed "s|^|$f\t|"
  done
}
nodes() {
  for n in dgx1 dgx2 dgx3 dgx4; do
    if [ "$n" = dgx1 ]; then bash -c "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$n" "$1"; fi
  done
}

say "watching for QUEUE job <UTC>-$spec.json started (queued at or after $since)"
nodes "rm -f $dir/STOP; mkdir -p $dir/out"
name=""; log=""
while [ -z "$name" ]; do
  while IFS=$'\t' read -r file text; do
    candidate=$(grep -oE "QUEUE job [0-9]{8}T[0-9]{6}-$spec\.json started" <<<"$text" | awk '{print $3}')
    [ -n "$candidate" ] || continue
    stamp=${candidate%%-*}
    if [[ "$stamp" > "$since" || "$stamp" == "$since" ]]; then name=$candidate; log=$file; fi
  done < <(new_lines)
  [ -n "$name" ] || sleep 1
done
say "queue started $name ($log); arming samplers"
nodes "sudo -n setsid -f python3 $dir/sampler.py --out $dir/out --node \$(hostname -s) --stop $dir/STOP \
  --max-seconds 18000 --rollup-seconds 5 --worker-seconds 60 --maps-seconds 30 < /dev/null >> $dir/sampler.log 2>&1"
sleep 2
nodes "echo \$(hostname -s): \$(pgrep -fc '[n]ext-memory/sampler.py --out') sampler process(es)"

until new_lines | grep -qE "QUEUE job $name (done|failed)"; do sleep 5; done
say "$(new_lines | grep -E "QUEUE job $name (done|failed)" | tail -1 | cut -f2)"
while true; do
  after=$(new_lines | grep -E "^$log" | cut -f2 | sed -n "/QUEUE job $name /,\$p")
  if grep -qE "QUEUE idle|QUEUE closed" <<<"$after"; then break; fi
  [ -e ~/spark-hold.json ] || break
  sleep 5
done
if [ -e ~/spark-hold.json ] && ! grep -q "QUEUE closed" <<<"$after"; then
  if [ -e ~/spark-request.json ]; then
    say "a request already exists: $(cat ~/spark-request.json)"
  else
    echo "{\"requester\": \"$me\", \"time\": \"$(date -u +%FT%TZ)\"}" > ~/spark-request.json
    say "requested the window back"
  fi
fi
while [ -e ~/spark-hold.json ]; do sleep 5; done
grep -q "next-memory (soak and start-up memory" ~/spark-request.json 2>/dev/null && rm ~/spark-request.json
say "hold gone; letting the restore boot settle ${settle}s"
sleep "$settle"
nodes "touch $dir/STOP; docker logs -t dsv41-karmic-kraken > $dir/out/engine-\$(hostname -s)-restore.log 2>&1; true"
sleep 5
for n in dgx2 dgx3 dgx4; do
  scp -q "$n:$dir/out/*" "$dir/out/" 2>/dev/null || say "copy from $n failed"
done
say "samplers stopped; outputs in $dir/out"
