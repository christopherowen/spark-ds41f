#!/usr/bin/env bash
# What late admission adds to startup memory, on one idle node (dgx4): the smoke test's
# dummy-weight server booted alternately with VLLM_LATE_ADMISSION_LAYER=-1 and 18, the
# candidate overlay mounted in both. For each boot: the lowest MemAvailable from launch to
# readiness (sampled every 0.1 s) and MemAvailable once ready. Run smoke.sh first (it writes
# the model directory).
set -u
LAB=${LAB:-$HOME/spark-lab/late-admission}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
D=$LAB/smoke
mounts=()
for f in $(cd "$LAB/candidate" && find vllm -type f -name '*.py'); do
  mounts+=(-v "$LAB/candidate/$f:$V/$f:ro")
done
avail() { awk '/^MemAvailable/{print $2}' /proc/meminfo; }

for layer in -1 18 -1 18; do
  docker rm -f la-mem > /dev/null 2>&1
  sync; sleep 5
  before=$(avail)
  low=$before
  docker run -d --name la-mem --gpus all --network host --shm-size 16g --memory 48g -w "$V" \
    "${mounts[@]}" -v "$D/tiny:/w/tiny:ro" \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_LATE_ADMISSION_LAYER="$layer" \
    --entrypoint python3 "$IMG" -m vllm.entrypoints.openai.api_server \
    --model /w/tiny --served-model-name tiny --load-format dummy --seed 0 \
    --async-scheduling --distributed-executor-backend mp --max-model-len 2048 \
    --max-num-seqs 8 --gpu-memory-utilization 0.25 --host 127.0.0.1 --port 8199 > /dev/null
  for _ in $(seq 6000); do
    now=$(avail)
    [ "$now" -lt "$low" ] && low=$now
    curl -sf -m 1 http://127.0.0.1:8199/v1/models > /dev/null && break
    sleep 0.1
  done
  sleep 5
  ready=$(avail)
  echo "layer $layer: before $((before / 1024)) MiB, startup low $((low / 1024)) MiB, ready $((ready / 1024)) MiB, startup drop $(((before - low) / 1024)) MiB, steady drop $(((before - ready) / 1024)) MiB"
  docker rm -f la-mem > /dev/null
done
