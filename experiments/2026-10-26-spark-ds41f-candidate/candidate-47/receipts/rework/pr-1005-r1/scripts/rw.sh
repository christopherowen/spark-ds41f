#!/bin/bash
# rw.sh LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
# ARM: an overlay dir under ovl/ (the 26 files 0762..0769 touch, bind-mounted read-only
# over /opt/spark-ds41f/candidate/vllm in spark-ds41f:1107), or "1107" (plain image).
# Takes /tmp/spark-lab-gpu.lock, records host state inside it, refuses to run when
# ~/spark-hold.json exists or another container runs; logs one block per run to out/commands.log.
W=$HOME/spark-lab/agents/candidate-47-rework-1005
IMG=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce
MODEL_HUB=$HOME/.cache/huggingface/hub
label=$1; arm=$2; shift 2
O=$W/out; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
if [ "$arm" != 1107 ]; then
  [ -d $W/ovl/$arm ] || { echo "no overlay $arm"; exit 2; }
  while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/changed-files.txt
fi
mkdir -p $W/c/tl-$arm
cmd=(docker run --rm --name "c47rw1005-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
     -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_L2_PREFETCH_GRID=2 -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_DS41_HEAD_DTYPE=float32
     -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$W/c/tl-$arm:/root/.tilelang" -v "$O:/out" -v "$W/scripts:/h:ro"
     -v "$MODEL_HUB:/hub:ro" -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Names}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
clk=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tr -d ' ')
{
  echo "### $label"
  echo "host: $(hostname) (sp3)"
  echo "arm: $arm $( [ "$arm" != 1107 ] && grep "^$arm = " $W/ovl/arms.txt | tail -1 | cut -d= -f2- )"
  echo "image: $IMG (spark-ds41f:1107)"
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "gpu_sm_clock_temp_before: $clk"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.log
if [ $hold = present ] || [ -n "$dps" ]; then printf 'skipped: hold file present or containers running\n\n' >> $O/commands.log; exit 3; fi
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: out/$label.txt"
  echo
} >> $O/commands.log
exit $rc
