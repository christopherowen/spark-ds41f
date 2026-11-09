#!/bin/bash
# om.sh PR LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
# ARM: 1107 = plain spark-ds41f:1107 image; c47 = same image with the 26 candidate-47
# files (0762..0769 applied to the image's own tree) bind-mounted read-only.
# Takes /tmp/spark-lab-gpu.lock, records host state inside the lock, refuses to run
# when ~/spark-hold.json exists, logs one block per run to out/pr-PR/commands.log.
B=$HOME/spark-lab/agents/candidate-47-own-marks
IMG=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce
pr=$1; label=$2; arm=$3; shift 3
O=$B/out/pr-$pr; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
if [ "$arm" = c47 ]; then
  while read -r f; do mounts+=(-v "$B/ovl/c47/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/changed-files.txt
elif [ "$arm" != 1107 ]; then echo "bad arm $arm"; exit 2; fi
mkdir -p $B/c/tl-$arm
cmd=(docker run --rm --name "c47om-$pr-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang" -v "$O:/out"
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Names}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
{
  echo "### $label"
  echo "host: $(hostname) ($(cat $B/alias))"
  echo "arm: $arm"
  if [ "$arm" = c47 ]; then echo "image: $IMG (spark-ds41f:1107) + candidate-47 overlay (26 files, ro; overlay-sha.txt)"; else echo "image: $IMG (spark-ds41f:1107)"; fi
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.log
if [ $hold = present ]; then printf 'skipped: hold file present\n\n' >> $O/commands.log; exit 3; fi
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: $label.txt"
  echo
} >> $O/commands.log
exit $rc
