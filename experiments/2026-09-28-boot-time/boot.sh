#!/bin/bash
# usage: boot.sh ARM LABEL [pyspy]   (on dgx1, deployment checkout at this experiment's commit)
# Stops every configuration, starts cluster-ARM.json with each launcher line
# timestamped, then saves every node's container log and inspect data under
# results/private/boot/LABEL and prints the timeline. With "pyspy", samples
# the dgx1 container's Python processes during the boot. The arm stays up.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-boot-time
arm=$1
label=$2
mode=${3:-}
out=results/private/boot/$label
mkdir -p "$out"
for config in config/cluster.json $E/cluster-*.json; do
  bin/spark3 --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
echo "$(date -u +%FT%T.%3NZ) start $arm as $label (dgx1 MemAvailable ${avail} GiB)"
sampler=
if [ "$mode" = pyspy ]; then
  python3 $E/pyspy_sampler.py "$out/pyspy.txt" &
  sampler=$!
fi
date -u +%FT%T.%3NZ > "$out/launch_t0"
bin/spark3 --cluster-config "$E/cluster-$arm.json" cluster start --replace --apply 2>&1 \
  | grep --line-buffered -v 'docker run' \
  | while IFS= read -r line; do echo "$(date -u +%FT%T.%3NZ) $line"; done | tee "$out/launcher.log" \
  | grep -E "launched|API ready|cluster ready|ERROR"
if [ -n "$sampler" ]; then kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null; fi
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker inspect -f '{{.State.StartedAt}}' dsv41-karmic-kraken" > "$out/$n.started" 2>/dev/null
  ssh -n "$n" "docker logs -t dsv41-karmic-kraken 2>&1" > "$out/$n.log" 2>/dev/null
done
python3 $E/timeline.py "$out"
