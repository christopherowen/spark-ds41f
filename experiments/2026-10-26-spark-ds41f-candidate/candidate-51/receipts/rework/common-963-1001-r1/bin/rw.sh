#!/bin/bash
# rw.sh PR LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
#
# One run of candidate-51's round-1 rework of #963 and #1001 on sp8, in a fresh container
# (a "boot"). Adapted from the own-mark run's om.sh (same arms, same mounts, same guards).
# ARM:
#   base  = spark-ds41f:candidate-50 (IMG) + candidate-50's four no1009/overlay files, ro
#           (tree 14c5a2dc as candidate-50/tp4-no1009.json mounts it);
#   c51   = the same image + the 15 files where candidate-51's tree (cc1115c5, tree 987ba058)
#           differs from the image's tree, ro (the own-mark run's c51 arm);
#   c51r1 = the same image + every file where the corrected candidate-51 tree
#           (0781, 0782 #963-r1, 0783, 0784 #1001-r1) differs from the image's tree, ro;
#   img   = the plain image.
# Takes /tmp/spark-lab-gpu.lock; inside it records loadavg, docker ps and the hold file,
# refuses to run when ~/spark-hold.json exists (exit 3) or a "vllm serve" container runs
# (exit 4), and logs one block per run to out/pr-PR/commands.txt.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001
IMG=$(cat $W/image-id)
pr=$1; label=$2; arm=$3; shift 3
O=$W/out/pr-$pr; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
case "$arm" in
  base|c51|c51r1) while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/files-$arm.txt ;;
  img)  ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
mkdir -p $W/c/tl-$arm
cmd=(docker run --rm --name "c51rw-$pr-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$W/c/tl-$arm:/root/.tilelang"
     -v "$O:/out" -v "$W/h:/h:ro"
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Names}} {{.Image}}' | tr '\n' ';')
serving=$(docker ps --no-trunc --format '{{.Names}} {{.Command}}' | grep -c 'vllm serve')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
{
  echo "### $label"
  echo "host: $(hostname) (sp8)"
  echo "arm: $arm"
  case "$arm" in
    base)  echo "image: $IMG (spark-ds41f:candidate-50) + candidate-50 no1009/overlay (4 files, ro; files-base.txt)" ;;
    c51)   echo "image: $IMG (spark-ds41f:candidate-50) + candidate-51 tree cc1115c5 files (15 files, ro; files-c51.txt)" ;;
    c51r1) echo "image: $IMG (spark-ds41f:candidate-50) + corrected candidate-51 tree $(cat $W/c51r1-head 2>/dev/null) files ($(wc -l < $W/files-c51r1.txt) files, ro; files-c51r1.txt)" ;;
    img)   echo "image: $IMG (spark-ds41f:candidate-50)" ;;
  esac
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.txt
if [ $hold = present ]; then printf 'skipped: hold file present\n\n' >> $O/commands.txt; exit 3; fi
if [ "$serving" != 0 ]; then printf 'skipped: a serving container is running\n\n' >> $O/commands.txt; exit 4; fi
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: $label.txt"
  echo
} >> $O/commands.txt
exit $rc
