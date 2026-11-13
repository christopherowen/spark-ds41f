#!/usr/bin/env bash
# usage: runtest.sh <label> <arm> <gpu|cpu>
# One pytest run of PR #979's own test files on this node, spark-ds41f:1107 with the arm's
# 15 touched files (7 vLLM sources + 8 tests) bound read-only over the image tree, under
# /tmp/spark-lab-gpu.lock. Refuses to run with ~/spark-hold.json present or a container
# running once the lock is held. Records command, host, image, UTC times, load, docker ps
# and exit code to out/tests-commands.txt; output to out/<label>.txt.
set -u
W=$HOME/spark-lab/agents/candidate-49-rework-979-r1
label=$1; arm=$2; kind=$3
IMG=spark-ds41f:1107
V=/opt/spark-ds41f/candidate/vllm
L=/tmp/spark-lab-gpu.lock
mounts=()
while read -r f; do mounts+=(-v "$W/ov/$arm/$f:$V/$f:ro"); done < "$W/touched.txt"
common=(--rm --name "c49rw979-$label" --entrypoint pytest -w /tmp
        -e PYTHONPATH=$V:/opt/spark-ds41f/candidate/b12x:/opt/release_tests
        -v "$W/release_tests:/opt/release_tests:ro" "${mounts[@]}")
if [ "$kind" = gpu ]; then
  cmd=(docker run --gpus all --ipc=host "${common[@]}" "$IMG" -p tblib_stub -v
       $V/tests/models/test_deepseek_v41_sp_project_reduce_scatter.py $V/tests/models/test_deepseek_v4_1_sp_prefill.py)
else
  cmd=(docker run --gpus all -e CUDA_VISIBLE_DEVICES= "${common[@]}" "$IMG" -p tblib_stub -v
       $V/tests/models/test_deepseek_v4_1_sp_prefill.py $V/tests/models/test_deepseek_v4_1_engram.py)
fi
C=$W/out/tests-commands.txt
{ echo "### $label"; echo "host: $(hostname)"; echo "image: $(docker image inspect --format '{{.Id}}' $IMG) ($IMG)";
  echo "lock_request_utc: $(date -u +%FT%TZ)"; } >> "$C"
exec 9>"$L"; flock 9
{
  echo "lock_held_utc: $(date -u +%FT%TZ)"
  echo "loadavg_before: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "docker_ps_before: [$(docker ps --format '{{.Names}}' | tr '\n' ' ')]"
  [ -e ~/spark-hold.json ] && echo "hold: PRESENT" || echo "hold: none"
  printf 'command: flock %s' "$L"; printf ' %q' "${cmd[@]}"; echo
} >> "$C"
if [ -e ~/spark-hold.json ] || [ -n "$(docker ps -q)" ]; then echo "ABORTED: hold or running container" >> "$C"; exit 98; fi
"${cmd[@]}" > "$W/out/$label.txt" 2>&1
rc=$?
{ echo "end_utc: $(date -u +%FT%TZ)"; echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)"; echo "exit_code: $rc"; echo "output: $label.txt"; echo; } >> "$C"
echo "rc=$rc $(tail -n 1 "$W/out/$label.txt")"
