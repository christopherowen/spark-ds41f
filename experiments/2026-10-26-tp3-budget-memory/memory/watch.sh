#!/usr/bin/env bash
# The TP3 budget sweep's and memory-cuts check's dgx1 watcher (from experiments/2026-10-23-next-release/memory/watch.sh,
# for the triangle), started before the sweep is queued:
#  1. when a queue runner starts <UTC>-$SPEC.json (queued at or after this
#     watcher started), arm the read-only samplers on dgx1-3 and a follower of every
#     serving container's engine log (`docker logs -f -t`: each arm's container is replaced
#     by the next arm's, and the workspace slot log and start-up lines live there);
#  2. when the queue idles after that job, ask for the window back (the runner restores
#     production);
#  3. once the hold is gone and the restore boot has settled, stop the samplers and the
#     followers, and copy every node's samples, engine logs and allocator traces to dgx1.
# Reads only queue-log bytes written after it started. Run it from ~/spark-lab/tp3-budget-memory,
# where sampler.py was copied on every node:
#   for n in dgx1 dgx2 dgx3; do ssh $n mkdir -p spark-lab/tp3-budget-memory; \
#     scp experiments/2026-10-26-tp3-budget-memory/memory/sampler.py $n:spark-lab/tp3-budget-memory/; done
#   cd ~/spark-lab/tp3-budget-memory && setsid -f bash ~/projects/spark-ds41f/experiments/2026-10-26-tp3-budget-memory/memory/watch.sh >> watch.log 2>&1
set -uo pipefail
spec=${SPEC:-tp3-budget-memory}  # the spec this window runs (tp3-memory-cuts: SPEC=tp3-memory-cuts)
dir=$HOME/spark-lab/tp3-budget-memory
logs=$HOME/projects/spark-ds41f/results/private/lab
traces=projects/spark-ds41f/cache/tp3-budget-memory
me="tp3-budget-memory (budget sweep memory attribution, dgx1 watcher): queue idle after my spec, releasing"
settle=${SETTLE_SECONDS:-240}
since=${SINCE:-$(date -u +%Y%m%dT%H%M%S)}
say() { echo "$(date -u +%FT%TZ) $*"; }
declare -A off
for f in "$logs"/*.log; do off[$f]=$(stat -c %s "$f"); done
new_lines() {  # every queue-log line appended since the watcher started (a new log from its start)
  for f in "$logs"/*.log; do
    tail -c +$(( ${off[$f]:-0} + 1 )) "$f" | sed "s|^|$f\t|"
  done
}
nodes() {
  for n in dgx1 dgx2 dgx3; do
    if [ "$n" = dgx1 ]; then bash -c "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$n" "$1"; fi
  done
}
# One follower per node: a `docker logs -f -t` per serving container it has not followed
# yet (candidate arms run as spark-ds41f, the restored 1022 as dsv41-karmic-kraken).
follower='while [ ! -e '"$dir"'/STOP ]; do
  for id in $(docker ps -q --no-trunc --filter name=^/spark-ds41f$ --filter name=^/dsv41-karmic-kraken$); do
    out='"$dir"'/out/engine-$(hostname -s)-${id:0:12}.log
    [ -e "$out" ] || setsid -f docker logs -f -t "$id" > "$out" 2>&1
  done
  sleep 2
done'

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
say "queue started $name ($log); arming samplers and engine-log followers"
nodes "sudo -n setsid -f python3 $dir/sampler.py --out $dir/out --node \$(hostname -s) --stop $dir/STOP \
  --max-seconds 18000 --rollup-seconds 5 --worker-seconds 30 --maps-seconds 30 < /dev/null >> $dir/sampler.log 2>&1"
nodes "setsid -f bash -c $(printf %q "$follower") < /dev/null >> $dir/follower.log 2>&1"
sleep 2
nodes "echo \$(hostname -s): \$(pgrep -fc '[t]p3-budget-memory/sampler.py --out') sampler processes"

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
grep -q "tp3-budget-memory (budget sweep memory attribution" ~/spark-request.json 2>/dev/null && rm ~/spark-request.json
say "hold gone; letting the restore boot settle ${settle}s"
sleep "$settle"
nodes "touch $dir/STOP; mkdir -p $dir/out/traces-\$(hostname -s); cp -r ~/$traces/. $dir/out/traces-\$(hostname -s)/ 2>/dev/null; true"
sleep 5
for n in dgx2 dgx3; do
  scp -q -r "$n:$dir/out/*" "$dir/out/" 2>/dev/null || say "copy from $n failed"
done
say "samplers and followers stopped; outputs in $dir/out"
