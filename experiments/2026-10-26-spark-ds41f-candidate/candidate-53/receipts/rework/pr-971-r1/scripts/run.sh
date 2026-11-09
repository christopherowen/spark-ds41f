#!/bin/bash
# run.sh LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]   (#971 round 1 rework, sp8)
# ARM: base = spark-ds41f:candidate-50 + candidate-50's 4 no1009/overlay files (ro), as the
#             candidate-53 own-mark run's base arm;
#      r1   = the image + the 32 files that differ between its tree (d5778ab5) and the corrected
#             candidate-53 tree (0781, 0782-r1, 0783, 0784, 0785; changed-files-r1.txt), ro;
#      orig = the image + the 33 files of the original candidate-53 tree (c0e25ff1), ro: the
#             own-mark run's c53 arm, byte-identical overlay.
# Takes /tmp/spark-lab-gpu.lock (or runs inside a q2.sh batch that holds it); inside the lock records host state and refuses (exit 3) when
# ~/spark-hold.json exists or any container is running. One block per run in out/commands.txt.
B=$HOME/spark-lab/agents/candidate-53-rework-971
IMG=$(cat $B/image-id)
label=$1; arm=$2; shift 2
O=$B/out; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
case $arm in
  r1|orig) while read -r f; do mounts+=(-v "$B/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/changed-files-$arm.txt;;
  base) while read -r f; do mounts+=(-v "$B/ovl/no1009/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/no1009-files.txt;;
  *) echo "bad arm $arm"; exit 2;;
esac
mkdir -p $B/c/tl-$arm
cmd=(docker run --rm --name "c53rw971-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang" -v "$O:/out"
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
# q2.sh takes the lock once for a batch of runs and sets SPARK_LAB_GPU_LOCK_HELD to the batch name.
if [ -z "$SPARK_LAB_GPU_LOCK_HELD" ]; then exec 9>>/tmp/spark-lab-gpu.lock; flock 9; fi
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
{
  echo "### $label"
  echo "host: $(hostname) (sp8)"
  echo "arm: $arm"
  case $arm in
    r1) echo "image: $IMG (spark-ds41f:candidate-50) + corrected candidate-53 overlay ($(wc -l < $B/changed-files-r1.txt) files, ro; overlay-sha.txt)";;
    orig) echo "image: $IMG (spark-ds41f:candidate-50) + original candidate-53 overlay ($(wc -l < $B/changed-files-orig.txt) files, ro; overlay-sha.txt)";;
    base) echo "image: $IMG (spark-ds41f:candidate-50) + no1009 overlay ($(wc -l < $B/no1009-files.txt) files, ro)";;
  esac
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  if [ -n "$SPARK_LAB_GPU_LOCK_HELD" ]; then echo "lock: /tmp/spark-lab-gpu.lock held by batch $SPARK_LAB_GPU_LOCK_HELD (q2.sh)"; fi
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
