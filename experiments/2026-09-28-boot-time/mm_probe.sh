#!/bin/bash
# usage: mm_probe.sh ARM LABEL   (on dgx1, deployment checkout at this experiment's commit)
# Boots ARM with boot.sh while sampling dgx1 MemAvailable, then runs the
# vision check twice (cold encoder, then warm) into results/private/boot/LABEL.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-boot-time
out=results/private/boot/$2
mkdir -p "$out"
( while sleep 0.5; do awk '/MemAvailable/{print $2}' /proc/meminfo; done ) > "$out/memavail.txt" &
sampler=$!
$E/boot.sh "$1" "$2"
kill $sampler
echo "dgx1 boot low point $(sort -n "$out/memavail.txt" | head -1 | awk '{printf "%.2f GiB", $1/1048576}')" | tee "$out/memlow.txt"
for pass in cold warm; do
  echo "== vision check ($pass)"
  python3 experiments/2026-09-25-vision/vision_check.py 2>&1 | tee "$out/vision-$pass.txt"
done
