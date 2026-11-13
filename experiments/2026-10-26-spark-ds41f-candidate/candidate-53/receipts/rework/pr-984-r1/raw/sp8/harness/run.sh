#!/bin/bash
# run.sh LABEL ARM [extra docker opts...]
# One tp3-switched rank-0 engine startup (boot_tp3r0.py) on sp8's GB10, in the 1107
# recipe's container settings, on spark-ds41f:candidate-50 (8c1ea1f6) with an arm's
# files bind-mounted read-only over /opt/spark-ds41f/candidate/vllm:
#   base  = candidate-50's 4 no1009/overlay files (candidate-50 without #1009);
#   c984  = the 22 non-test files of candidate-53 without #960 and #998 (c53nn/),
#           i.e. candidate-50-no1009 + #971 + #977 + #984 as submitted;
#   no984 = c53nn with #984's 8 files at their pre-#984 versions (no960-no998-no984);
#   fix   = c53nn with #984's files corrected (fix/).
# Takes /tmp/spark-lab-gpu.lock; inside it records host state and refuses to run when
# ~/spark-hold.json exists or any container runs (exit 3). Appends one block per run to
# out/commands.txt; the run's output goes to out/LABEL.log.
set -u
W=$HOME/spark-lab/agents/rework-984-r1
IMG=$(docker image inspect spark-ds41f:candidate-50 --format '{{.Id}}')
case $IMG in sha256:8c1ea1f6*) ;; *) echo "spark-ds41f:candidate-50 is $IMG, not 8c1ea1f6"; exit 2 ;; esac
label=$1; arm=$2; shift 2
O=$W/out; mkdir -p $O
case $arm in
  base) SRC=$W/main/experiments/2026-10-26-spark-ds41f-candidate/candidate-50/no1009/overlay ;;
  c984) SRC=$W/c53nn ;;
  no984) SRC=$W/no984 ;;
  fix) SRC=$W/fix ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
mounts=()
while read -r f; do
  mounts+=(-v "$SRC/$f:/opt/spark-ds41f/candidate/vllm/$f:ro")
done < <(cd $SRC && find vllm -type f -name '*.py' | sort)
H=/home/chris/.cache/huggingface/hub
GLP=$H/models--msuiche--DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75/snapshots/2715ffd51f59899115853e1118db080536966c44/glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf
cmd=(docker run --rm --name "rework984-$label" --network=host --ipc=host --gpus=all
     --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --ulimit=stack=67108864:67108864 --ulimit=core=0
     --security-opt=seccomp=unconfined --security-opt=label=disable --shm-size=16g
     --memory=112g --memory-swap=112g
     --mount=type=bind,source=/dev/dri/by-path/pci-000f:01:00.0-card,target=/dev/dri/card0
     --device-cgroup-rule='c 226:* rw' --cpuset-cpus=5-9,15-19
     --env-file $W/harness/env.list
     -v $H:/hub:ro -v $W/cache:/cache:rw -v $GLP:/glp/vector.gguf:ro
     -v $W/harness:/harness:ro -v $O:/out
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" "$@"
     --entrypoint python3 $IMG /harness/boot_tp3r0.py --label "$label" --out /out
     --serve-args "$(cat $W/harness/serve-args.json)")
issued=$(date -u +%FT%TZ)
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
start=$(date -u +%FT%TZ)
la=$(cut -d' ' -f1-3 /proc/loadavg)
dps=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | tr '\n' ';')
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
memavail=$(awk '/MemAvailable/{printf "%.2f GiB", $2/1048576}' /proc/meminfo)
{
  echo "### $label"
  echo "host: $(hostname) (sp8)"
  echo "arm: $arm ($(cd $SRC && find vllm -type f -name '*.py' | wc -l) files from ${SRC#$W/}, ro)"
  echo "image: $IMG (spark-ds41f:candidate-50)"
  echo "issued_utc: $issued"
  echo "start_utc: $start"
  echo "loadavg_before: $la"
  echo "memavailable_before: $memavail"
  echo "docker_ps_before: ${dps:-none}"
  echo "hold_file: $hold"
  echo "command: flock /tmp/spark-lab-gpu.lock $(printf '%q ' "${cmd[@]}")"
} >> $O/commands.txt
if [ $hold = present ]; then printf 'skipped: hold file present\n\n' >> $O/commands.txt; exit 3; fi
if [ -n "$dps" ]; then printf 'skipped: a container is running\n\n' >> $O/commands.txt; exit 3; fi
"${cmd[@]}" > "$O/$label.log" 2>&1; rc=$?
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $rc"
  echo "output: $label.log, $label.json, $label.memavail.txt"
  echo
} >> $O/commands.txt
echo "$label rc=$rc"
exit $rc
