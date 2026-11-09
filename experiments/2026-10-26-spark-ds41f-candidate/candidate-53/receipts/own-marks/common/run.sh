#!/bin/bash
# run.sh PR LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
# ARM: base = spark-ds41f:candidate-50 with candidate-50's 4 no1009/overlay files bind-mounted
#      read-only (as candidate-50/tp3-switched-no1009.json mounts them);
#      c53  = the same image with the 33 files that differ between its tree (d5778ab5) and
#      candidate-53's tree (c0e25ff1) bind-mounted read-only (changed-files.txt, ovl/c53).
# Takes /tmp/spark-lab-gpu.lock; inside the lock records host state and refuses to run when
# ~/spark-hold.json exists or any container is running (exit 3). Logs one block per run to
# out/pr-PR/commands.txt and the combined output to out/pr-PR/LABEL.txt.
B=$HOME/spark-lab/agents/candidate-53-own-marks
IMG=$(cat $B/image-id)
pr=$1; label=$2; arm=$3; shift 3
O=$B/out/pr-$pr; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
if [ "$arm" = c53 ]; then
  while read -r f; do mounts+=(-v "$B/ovl/c53/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/changed-files.txt
elif [ "$arm" = base ]; then
  while read -r f; do mounts+=(-v "$B/ovl/no1009/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/no1009-files.txt
else echo "bad arm $arm"; exit 2; fi
mkdir -p $B/c/tl-$arm
cmd=(docker run --rm --name "c53om-$pr-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang" -v "$O:/out"
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
{
  echo "### $label"
  echo "host: $(hostname) (sp8)"
  echo "arm: $arm"
  if [ "$arm" = c53 ]; then echo "image: $IMG (spark-ds41f:candidate-50) + candidate-53 overlay ($(wc -l < $B/changed-files.txt) files, ro; overlay-sha.txt)";
  else echo "image: $IMG (spark-ds41f:candidate-50) + no1009 overlay ($(wc -l < $B/no1009-files.txt) files, ro)"; fi
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.txt
if [ $hold = present ]; then printf 'skipped: hold file present\n\n' >> $O/commands.txt; exit 3; fi
if [ -n "$dps" ]; then printf 'skipped: a container is running\n\n' >> $O/commands.txt; exit 3; fi
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: $label.txt"
  echo
} >> $O/commands.txt
echo "$label rc=$rc"
exit $rc
