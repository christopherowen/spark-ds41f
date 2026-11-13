#!/bin/bash
# usage: run.sh ARM NAME [docker-extra-args...] -- CMD...
# ARM: base (candidate-50 + no1009), c53 (candidate-53 tree files), fix (candidate-53 + corrected 0785)
set -u
D=$HOME/spark-lab/agents/candidate-53-rework-998
IMAGE=spark-ds41f:candidate-50
ROOT=/opt/spark-ds41f/candidate/vllm
arm=$1; name=$2; shift 2
extra=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
shift
mounts=()
case $arm in
  base) for f in $(cd $D/no1009 && find vllm -type f | sort); do mounts+=(-v "$D/no1009/$f:$ROOT/$f:ro"); done ;;
  c53)  for f in $(cat $D/src-diff.txt); do mounts+=(-v "$D/vllm-c53/$f:$ROOT/$f:ro"); done ;;
  fix)  for f in $(cat $D/src-diff.txt); do mounts+=(-v "$D/vllm-fix/$f:$ROOT/$f:ro"); done ;;
  *) echo "bad arm"; exit 2 ;;
esac
if [ -e $HOME/spark-hold.json ]; then echo "HOLD FILE PRESENT: refusing"; cat $HOME/spark-hold.json; exit 3; fi
serving=$(docker ps --format '{{.Names}} {{.Image}}' | grep -Ei 'serve|vllm-server|spark-ds41f-(tp|rank|head)' || true)
if [ -n "$serving" ]; then echo "SERVING CONTAINER RUNNING: refusing: $serving"; exit 4; fi
out=$D/out/$name.txt
mkdir -p $D/out $D/tlcache
cname=c53rw998-$name
cmd=(flock /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host --name "$cname" --device /dev/dri/card0
     -e TILELANG_CACHE_DIR=/c/tl -v $D/tlcache:/c/tl -e HF_HUB_OFFLINE=1
     --entrypoint '' -w $ROOT "${mounts[@]}" "${extra[@]}" $IMAGE "$@")
{
  echo "### $name"
  echo "host: $(hostname) arm: $arm"
  echo "image: $(docker image inspect $IMAGE --format '{{.Id}}') ($IMAGE)"
  echo "overlay: $arm ($(echo ${#mounts[@]}/2 | bc) files)"
  echo "start_utc: $(date -u +%FT%TZ)"
  echo "hold_file: absent; docker_ps_before: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  echo "command: ${cmd[*]//$D/\$D}" | sed "s#-v \$D/[^ ]*:$ROOT/[^ ]*:ro ##g"
  echo "mounted_files: $arm overlay over $ROOT"
} >> $D/out/commands.txt
start=$(date +%s)
"${cmd[@]}" > $out 2>&1
rc=$?
echo "exit: $rc  wall_s: $(( $(date +%s) - start ))  output: out/$name.txt" >> $D/out/commands.txt
echo >> $D/out/commands.txt
echo "rc=$rc"; tail -5 $out
exit $rc
