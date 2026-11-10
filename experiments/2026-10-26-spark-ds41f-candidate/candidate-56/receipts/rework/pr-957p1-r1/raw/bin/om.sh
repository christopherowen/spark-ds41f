#!/bin/bash
# om.sh LABEL ARM [docker-opts...] -- ENTRYPOINT [args...]
#
# One run of the #957 part 1 round 1 rework on sp8, in a fresh container (a "boot"),
# round 3's own-mark runner (round3-own-marks/bin/om.sh) with this rework's arms:
#   base = spark-ds41f:candidate-54 (tree 5f021d9a);
#   r1   = the same image + 0795-r1's files (files-r1.txt), read-only;
#   c56r1 = spark-ds41f:candidate-56 (image-id-c56) + candidate-54 + 0795-r1 + 0796's files
#           (files-c56r1.txt), read-only: the testbed profile's code, for its tests.
# Takes /tmp/spark-lab-gpu.lock itself (do not wrap it); inside the lock it records loadavg,
# MemAvailable, docker ps and the hold file, refuses to run when ~/spark-hold.json exists
# (exit 3) or a "vllm serve" container runs (exit 4), and logs one block per run to
# out/commands.txt.
W=$HOME/spark-lab/agents/candidate-56-rework-957p1
IMG=$(cat $W/image-id)
label=$1; arm=$2; shift 2
O=$W/out; mkdir -p $O
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
ep=$1; shift
mounts=()
case "$arm" in
  base) ;;
  r1|c56r1) while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/files-$arm.txt ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
[ $arm = c56r1 ] && IMG=$(cat $W/image-id-c56)
tag=$([ $arm = c56r1 ] && echo candidate-56 || echo candidate-54)
mkdir -p $W/c/tl-$arm
cmd=(docker run --rm --name "rw957r1-$label" --gpus all --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined
     -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
     -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h
     -e TILELANG_CACHE_DIR=/root/.tilelang/cache -v "$W/c/tl-$arm:/root/.tilelang"
     -v "$O:/out" -v "$W/h:/h:ro" -v "$W/lists:/lists:ro"
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
  echo "image: $IMG (spark-ds41f:$tag)$( [ $arm != base ] && echo " + files-$arm.txt ($(wc -l < $W/files-$arm.txt) files, ro)")"
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
    if docker ps --format '{{.Names}}' | grep -qx "rw957r1-$label"; then seen=1; elif [ $seen = 1 ]; then break; fi
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt "$lowest" ]; then lowest=$a; echo "$lowest" > "$O/.$label.minmem"; fi
    if [ "$a" -lt 6 ]; then docker kill "rw957r1-$label" >/dev/null 2>&1; echo "memguard: killed at ${a} GiB" >> "$O/$label.memguard"; break; fi
  done ) &
guard=$!
"${cmd[@]}" > "$O/logs/$label.txt" 2>&1; rc=$?
kill $guard 2>/dev/null; wait $guard 2>/dev/null
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  [ -f "$O/.$label.minmem" ] && echo "memavailable_lowest_during: $(cat "$O/.$label.minmem") GiB" && rm -f "$O/.$label.minmem"
  echo "output: logs/$label.txt"
  echo
} >> $O/commands.txt
exit $rc
