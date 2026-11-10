#!/bin/bash
# run.sh LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
# ARM: base = spark-ds41f:candidate-50 + candidate-54's 36 differing files (ovl/base, ro)
#      cand = the same image + the same 36 paths from the guard commit's tree (ovl/cand, ro)
# Takes /tmp/spark-lab-gpu.lock; inside the lock records host state and refuses to run (exit 3)
# when ~/spark-hold.json exists or any container is running. Logs one block per run to
# out/commands.txt and the combined output to out/LABEL.txt.
W=$HOME/spark-lab/agents/round3-995-guard
IMG=$(cat $W/image-id)
label=$1; arm=$2; shift 2
O=$W/out; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
case $arm in base|cand) ;; *) echo "bad arm $arm"; exit 2;; esac
mounts=()
while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/files.txt
mkdir -p $W/c/tl-$arm
cmd=(docker run --rm --name "r3g995-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$W/c/tl-$arm:/root/.tilelang" -v "$O:/out"
     -v "$W/harness:/h:ro" -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}"
     --entrypoint "$ep" "$IMG" "$@")
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
  if [ "$arm" = base ]; then echo "image: $IMG (spark-ds41f:candidate-50) + candidate-54 overlay ($(wc -l < $W/files.txt) files, ro; overlay-sha.txt)";
  else echo "image: $IMG (spark-ds41f:candidate-50) + candidate-54 + 0781 guard overlay ($(wc -l < $W/files.txt) files, ro; overlay-sha.txt)"; fi
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
