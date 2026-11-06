#!/bin/bash
set -euo pipefail
# One bounded, exclusive kernel job. Called only by the hold-owning controller.
checkout=/home/swank/spark-lab/l2-access-retention/checkout
run_root=/home/swank/spark-lab/l2-access-retention
boot=$1
revision=$2
holder=$3
[[ "$boot" =~ ^[12]$ ]]
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
experiment=experiments/2026-11-06-l2-access-retention
image=sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b
name=l2-access-retention-boot-$boot
out=$run_root/boot-$boot-$revision
test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$checkout" status --porcelain)"
ssh -n dgx1 cat /home/swank/spark-hold.json | python3 -c 'import json,sys; assert json.load(sys.stdin)["holder"]==sys.argv[1]' "$holder"
exec 9>/tmp/spark-lab-gpu.lock
flock -n 9
test -z "$(docker ps -q)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -gt 25165824
mkdir "$out"
mkdir -p "$out/cache"
# Compile the exact published source before cooling and GPU measurements.
flock -n /tmp/spark-cpu-analysis.lock taskset -c 0,1 \
  /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_121 \
  "$checkout/$experiment/capability.cu" -o "$out/capability" \
  > "$out/capability-build.txt" 2>&1
sha256sum "$checkout/$experiment/capability.cu" "$out/capability" > "$out/capability-hashes.txt"
systemctl cat dgx-fan-control >/dev/null
fan_before=$(systemctl is-active dgx-fan-control || true)
printf '%s\n' "$fan_before" > "$out/fan-before.txt"
cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  if test "$fan_before" = active; then
    sudo -n systemctl start dgx-fan-control
  else
    sudo -n systemctl stop dgx-fan-control
  fi
  systemctl is-active dgx-fan-control > "$out/fan-after.txt" || true
}
trap cleanup EXIT
thermal() {
  python3 -c 'from pathlib import Path; print(max(int(p.read_text()) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")))'
}
thermal > "$out/thermal-before.txt"
if test "$(thermal)" -ge 55000; then
  sudo -n systemctl stop dgx-fan-control
  sudo -n dgx-fan-control set-state 12
  for attempt in $(seq 1 60); do
    if test "$(thermal)" -lt 55000; then break; fi
    sleep 5
  done
fi
test "$(thermal)" -lt 55000
sudo -n systemctl reset-failed dgx-fan-control
sudo -n systemctl start dgx-fan-control
systemctl is-active --quiet dgx-fan-control
thermal > "$out/thermal-start.txt"
cat /proc/meminfo > "$out/memory-before.txt"
docker image inspect "$image" --format '{{.Id}}' > "$out/image.txt"
git -C "$checkout" rev-parse HEAD > "$out/revision.txt"
timeout --signal=TERM --kill-after=5 30 "$out/capability" \
  > "$out/capability.jsonl" 2> "$out/capability-errors.txt"
checkpoint=/home/swank/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
set +e
timeout --signal=TERM --kill-after=20 480 docker run --rm --name "$name" \
  --gpus all --network none --cpus=2 --memory=8g --memory-swap=8g \
  --ulimit core=0 --shm-size=1g \
  -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 -e TILELANG_COMPILE_THREADS=2 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e CUDA_CACHE_PATH=/cache/cuda \
  -v "$checkout:/experiment:ro" \
  -v /home/swank/.cache/huggingface/hub:/home/swank/.cache/huggingface/hub:ro \
  -v "$out:/out" -v "$out/cache:/cache" --entrypoint python3 "$image" \
  "/experiment/$experiment/markov_probe.py" --checkpoint "$checkpoint" \
  --boot "$boot" --output /out/measurements.json > "$out/output.txt" 2>&1 &
job=$!
while kill -0 "$job" 2>/dev/null; do
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  temp=$(thermal)
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$available" "$temp" >> "$out/telemetry.txt"
  if test "$available" -lt 16777216 || test "$temp" -ge 85000; then
    printf '%s\n' 'memory or thermal guard stop' >> "$out/guard.txt"
    docker stop -t 5 "$name" >/dev/null 2>&1
    break
  fi
  sleep 2
done
wait "$job"
status=$?
set -e
if test -f "$out/guard.txt"; then status=90; fi
cat /proc/meminfo > "$out/memory-after.txt"
thermal > "$out/thermal-after.txt"
printf '%s\n' "$status" > "$out/exit.txt"
exit "$status"
