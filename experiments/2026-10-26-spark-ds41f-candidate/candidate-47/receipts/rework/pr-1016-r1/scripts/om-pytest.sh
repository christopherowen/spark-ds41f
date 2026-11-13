#!/bin/bash
# om-pytest.sh LABEL ARM (om-r1.sh with the repo release-test pytest plugins at /rt) -- ENTRYPOINT [args...]
# ARM: 1107 = plain spark-ds41f:1107; no1016 / c47 / r1 = the same image with that overlay's
# 26 files (setup_r1.sh) bind-mounted read-only over /opt/spark-ds41f/candidate/vllm.
# Takes /tmp/spark-lab-gpu.lock, records host state inside the lock, refuses to run when
# ~/spark-hold.json exists or a container is running, logs one block per run to out/commands.log.
B=$HOME/spark-lab/agents/candidate-47-rework-1016
IMG=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce
label=$1; arm=$2; shift 2
[ "$1" = "--" ] && shift
ep=$1; shift
O=$B/out; mkdir -p $O
mounts=()
case $arm in
  1107) ;;
  no1016|c47|r1) while read -r f; do mounts+=(-v "$B/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/changed-files.txt ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
mkdir -p $B/c/tl-$arm
cmd=(docker run --rm --name "c47r1016-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang" -v "$O:/out"
     -v "$B/src/repo/scripts/release_tests:/rt:ro" -e PYTHONPATH=/rt:/opt/spark-ds41f/candidate/vllm -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}"
     --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Names}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
temp=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr -d ' ')
{
  echo "### $label"
  echo "host: $(hostname) (sp4)"
  echo "arm: $arm"
  if [ "$arm" = 1107 ]; then echo "image: $IMG (spark-ds41f:1107)"; else echo "image: $IMG (spark-ds41f:1107) + overlay $arm (26 files, ro; overlay-sha-$arm.txt)"; fi
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "gpu_temp_c,sm_clock_before: $temp"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.log
if [ $hold = present ] || [ -n "$dps" ]; then printf 'skipped: hold file or running container\n\n' >> $O/commands.log; exit 3; fi
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: $label.txt"
  echo
} >> $O/commands.log
exit $rc
