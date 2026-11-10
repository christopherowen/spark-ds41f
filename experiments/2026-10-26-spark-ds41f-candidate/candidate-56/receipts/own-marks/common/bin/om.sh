#!/bin/bash
# om.sh PR LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
#
# One own-mark run of round 3 (candidates 55-57) on sp8, in a fresh container (a "boot").
# Every arm is spark-ds41f:candidate-54 (IMG, tree 5f021d9a):
#   base  = the plain image (candidate-54's tree);
#   c56p1 = the image + 0795's 3 files (957 part 1) at e4b134ab, read-only;
#   c57p2 = the image + 0797's file (957 part 2) at 4fc8de11, read-only;
#   c57p4 = the image + 0798's 6 files (957 part 4) at d74e38ee, read-only;
#   c55dr = the image + 0794's 9 files (#988 dead-row part) at 410463b3, read-only.
# Each candidate arm is candidate-54 plus that one patch (files-ARM.txt; lists/ARM.txt is
# the expected tree). Takes /tmp/spark-lab-gpu.lock; inside the lock it records loadavg,
# MemAvailable, docker ps and the hold file, refuses to run when ~/spark-hold.json exists
# (exit 3) or a "vllm serve" container runs (exit 4), and logs one block per run to
# out/pr-PR/commands.txt.
B=$HOME/spark-lab/agents/round3-own-marks
IMG=$(cat $B/image-id)
pr=$1; label=$2; arm=$3; shift 3
O=$B/out/pr-$pr; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
case "$arm" in
  base) ;;
  c56p1|c57p2|c57p4|c55dr)
    while read -r f; do mounts+=(-v "$B/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/files-$arm.txt ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
mkdir -p $B/c/tl-$arm
cmd=(docker run --rm --name "r3om-$pr-$label" --gpus all --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined
     -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
     -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$B/c/tl-$arm:/root/.tilelang"
     -v "$O:/out" -v "$B/h:/h:ro" -v "$B/src:/src:ro" -v "$B/lists:/lists:ro"
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "${opts[@]}" --entrypoint "$ep" "$IMG" "$@")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
mem=$(awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)
dps=$(docker ps --format '{{.ID}} {{.Names}} {{.Image}}' | tr '\n' ';')
serving=$(docker ps --no-trunc --format '{{.Names}} {{.Command}}' | grep -c 'vllm serve')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
{
  echo "### $label"
  echo "host: $(hostname) (sp8)"
  echo "arm: $arm"
  echo "image: $IMG (spark-ds41f:candidate-54)$( [ $arm != base ] && echo " + files-$arm.txt ($(wc -l < $B/files-$arm.txt) files, ro)")"
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "memavailable_before: $mem"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.txt
if [ $hold = present ]; then printf 'skipped: hold file present\n\n' >> $O/commands.txt; exit 3; fi
if [ "$serving" != 0 ]; then printf 'skipped: a serving container is running\n\n' >> $O/commands.txt; exit 4; fi
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ -n "${MINMEM_GIB:-}" ] && [ "$avail" -lt "$MINMEM_GIB" ]; then
  printf 'skipped: MemAvailable %s GiB under %s GiB\n\n' "$avail" "$MINMEM_GIB" >> $O/commands.txt; exit 5
fi
# Memory guard: stop only this run's own container if MemAvailable falls under 6 GiB.
( seen=0; lowest=999
  while sleep 2; do
    if docker ps --format '{{.Names}}' | grep -qx "r3om-$pr-$label"; then seen=1; elif [ $seen = 1 ]; then break; fi
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt "$lowest" ]; then lowest=$a; echo "$lowest" > "$O/.$label.minmem"; fi
    if [ "$a" -lt 6 ]; then docker kill "r3om-$pr-$label" >/dev/null 2>&1; echo "memguard: killed at ${a} GiB" >> "$O/$label.memguard"; break; fi
  done ) &
guard=$!
"${cmd[@]}" > "$O/$label.txt" 2>&1; rc=$?
kill $guard 2>/dev/null; wait $guard 2>/dev/null
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  [ -f "$O/.$label.minmem" ] && echo "memavailable_lowest_during: $(cat "$O/.$label.minmem") GiB" && rm -f "$O/.$label.minmem"
  echo "output: $label.txt"
  echo
} >> $O/commands.txt
exit $rc
