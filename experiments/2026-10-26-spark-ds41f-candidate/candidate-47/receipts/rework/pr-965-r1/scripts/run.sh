#!/bin/bash
# usage: run.sh <label> <overlay> [--] <entrypoint> <args...>
# Runs <entrypoint> in spark-ds41f:1107 with ovl/<overlay>/{vllm,tests}/... bind-mounted read-only
# over /opt/spark-ds41f/candidate/vllm/, under the shared GPU lock. Appends a block to
# commands.txt and writes the output to out/<label>.txt.
W=$HOME/spark-lab/agents/candidate-47-rework-965
cd $W
LABEL=$1; OVL=$2; shift 2; [ "$1" = "--" ] && shift
ENTRY=$1; shift
mkdir -p out
HOLD=$(cat ~/spark-hold.json 2>/dev/null | tr -d '\n')
if [ -n "$HOLD" ]; then echo "hold file present: $HOLD"; exit 99; fi
IMG=$(docker image inspect spark-ds41f:1107 --format '{{.Id}}')
MOUNTS=()
if [ "$OVL" != "none" ]; then
  while IFS= read -r f; do
    MOUNTS+=(-v "$W/ovl/$OVL/$f:/opt/spark-ds41f/candidate/vllm/$f:ro")
  done < <(cd ovl/$OVL && find vllm tests -type f | sort)
fi
EXTRA=(-v "$W/c:/c" -v "$W/src/release_tests:/opt/release_tests:ro" -v "$W/src/harness:/harness:ro" -v "$W/out:/out")
CMD=(flock /tmp/spark-lab-gpu.lock docker run --rm --name c47rw-965-$LABEL --gpus all --ipc=host
     --entrypoint "$ENTRY" -w /opt/spark-ds41f/candidate/vllm
     -e HF_HUB_OFFLINE=1 -e TILELANG_CACHE_DIR=/c/tilelang
     -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x:/opt/release_tests
     "${MOUNTS[@]}" "${EXTRA[@]}" "$IMG" "$@")
{
  echo "### $LABEL"
  echo "host: $(hostname) (sp8)"
  echo "image: $IMG (spark-ds41f:1107)"
  echo "overlay: ovl/$OVL ($( [ "$OVL" = none ] && echo 0 || (cd ovl/$OVL && find vllm tests -type f | wc -l)) files, read-only over /opt/spark-ds41f/candidate/vllm)"
  echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "loadavg_before: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "docker_ps_before: $(docker ps --format '{{.Image}} {{.Names}};' | tr '\n' ' ' | sed 's/ $//')"
  echo "hold_file: ${HOLD:-none}"
  echo "command: ${CMD[*]}"
} >> commands.txt
"${CMD[@]}" > out/$LABEL.txt 2>&1
RC=$?
{
  echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "exit_code: $RC"
  echo "output: out/$LABEL.txt"
  echo
} >> commands.txt
exit $RC
