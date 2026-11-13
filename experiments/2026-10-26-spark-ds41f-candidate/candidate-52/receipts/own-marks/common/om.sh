#!/bin/bash
# om.sh PR LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
# ARM: base = spark-ds41f:candidate-50 with candidate-50's 4 no1009/overlay files bind-mounted read-only
#             (as candidate-50/tp4-no1009.json mounts them): candidate-50 without #1009, candidate-52's base.
#      c52  = the same image with the 13 files in which candidate-52's tree (51a16b66, tree 1116740c) differs
#             from the image's (candidate-50, ba7400d9, tree d5778ab5) bind-mounted read-only, the no1009 files
#             included (changed-files.txt, overlay-sha.txt).
# Takes /tmp/spark-lab-gpu.lock, then records host state inside the lock; refuses to run when
# ~/spark-hold.json exists or a serving container runs. One block per run in out/pr-PR/commands.log.
# With OM_LOCK_HELD=1 the caller already holds the lock (queue.sh run as `flock /tmp/spark-lab-gpu.lock
# bash queue.sh <section>`, so a section's ABBA runs back to back with no other job between); om.sh then
# does not take it again (a second flock from a child would wait on its own parent).
B=$HOME/spark-lab/agents/candidate-52-own-marks
IMG=$(cat $B/image-id)
pr=$1; label=$2; arm=$3; shift 3
O=$B/out/pr-$pr; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
case $arm in
  base) while read -r f; do mounts+=(-v "$B/ovl/no1009/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/no1009-files.txt;;
  c52)  while read -r f; do mounts+=(-v "$B/ovl/c52/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/changed-files.txt;;
  *) echo "bad arm $arm"; exit 2;;
esac
mkdir -p $B/c/tl-$arm
if [ -e "$O/$label.txt" ]; then echo "om.sh: $O/$label.txt exists, not re-running $label"; exit 4; fi
cmd=(docker run --rm --name "c52om-$pr-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang" -v "$O:/out"
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
if [ "${OM_LOCK_HELD:-0}" = 1 ]; then lockinfo="held by the enclosing queue section (flock /tmp/spark-lab-gpu.lock bash queue.sh)"
else exec 9>>/tmp/spark-lab-gpu.lock; flock 9; lockinfo="taken by om.sh"; fi
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Names}} {{.Command}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
if docker ps --format '{{.Command}}' --no-trunc | grep -q 'vllm serve\|"serve"'; then serving=yes; else serving=no; fi
{
  echo "### $label"
  echo "host: $(hostname) (sp8)"
  echo "arm: $arm"
  if [ "$arm" = c52 ]; then echo "image: $IMG (spark-ds41f:candidate-50) + candidate-52 overlay (13 files, ro; overlay-sha.txt)";
  else echo "image: $IMG (spark-ds41f:candidate-50) + no1009 overlay (4 files, ro)"; fi
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "serving_container: $serving"
  echo "gpu_lock: $lockinfo"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.log
if [ $hold = present ] || [ $serving = yes ]; then printf 'skipped: hold file or serving container present\n\n' >> $O/commands.log; exit 3; fi
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: $label.txt"
  echo
} >> $O/commands.log
exit $rc
