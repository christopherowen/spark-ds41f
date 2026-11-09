#!/bin/bash
# om.sh PR LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
#
# One own-mark run of candidate-51 on sp8, in a fresh container (a "boot").
# ARM:
#   base = spark-ds41f:candidate-50 (IMG) with candidate-50's four no1009/overlay files
#          bind-mounted read-only, as candidate-50/tp4-no1009.json mounts them;
#   c51  = the same image with the 15 files in which candidate-51's tree (cc1115c5, tree
#          987ba058) differs from the image's tree (candidate-50, ba7400d9, tree d5778ab5)
#          bind-mounted read-only (the 4 no1009 files and 2 no1009 test files among them);
#   img  = the plain image (tree verification only).
# Takes /tmp/spark-lab-gpu.lock; inside the lock it records loadavg, docker ps and the
# hold file, refuses to run when ~/spark-hold.json exists (exit 3) or a serving container
# ("vllm serve") is running (exit 4), and logs one block per run to out/pr-PR/commands.txt.
B=$HOME/spark-lab/agents/candidate-51-own-marks
IMG=$(cat $B/image-id)
pr=$1; label=$2; arm=$3; shift 3
O=$B/out/pr-$pr; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
case "$arm" in
  base) while read -r f; do mounts+=(-v "$B/ovl/base/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/files-base.txt ;;
  c51)  while read -r f; do mounts+=(-v "$B/ovl/c51/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/files-c51.txt ;;
  img)  ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
mkdir -p $B/c/tl-$arm
cmd=(docker run --rm --name "c51om-$pr-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang"
     -v "$O:/out" -v "$B/h:/h:ro" -v "$B/src:/src:ro"
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
    base) echo "image: $IMG (spark-ds41f:candidate-50) + candidate-50 no1009/overlay (4 files, ro; files-base.txt)" ;;
    c51)  echo "image: $IMG (spark-ds41f:candidate-50) + candidate-51 tree cc1115c5 files (15 files, ro; files-c51.txt)" ;;
    img)  echo "image: $IMG (spark-ds41f:candidate-50)" ;;
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
