#!/bin/bash
# pytest.sh LABEL: the corrected #984 tests and the files around them, on
# spark-ds41f:candidate-50 with fix/ (22 non-test files) and the corrected test files
# mounted read-only, as candidate-53 own marks ran #984 (PYTHONPATH with
# tests/vllm_test_utils and the tblib stub). Same lock and guards as run.sh.
set -u
W=$HOME/spark-lab/agents/rework-984-r1
IMG=$(docker image inspect spark-ds41f:candidate-50 --format "{{.Id}}")
case $IMG in sha256:8c1ea1f6*) ;; *) echo "bad image $IMG"; exit 2 ;; esac
label=$1; O=$W/out; mkdir -p $O
mounts=()
while read -r f; do mounts+=(-v "$W/fix/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < <(cd $W/fix && find vllm -type f -name "*.py" | sort)
while read -r f; do mounts+=(-v "$W/fixtests/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < <(cd $W/fixtests && find tests -type f | sort)
STUB=$HOME/spark-lab/agents/candidate-53-own-marks/stub_pkgs
cmd=(docker run --rm --name "rework984-$label" --gpus all --ipc=host -e HF_HUB_OFFLINE=1
     --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v $STUB:/tmp/stub_pkgs:ro
     -e PYTHONPATH=/tmp/stub_pkgs:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/vllm/tests/vllm_test_utils
     -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" --entrypoint pytest $IMG
     -v -k "not test_memory_profiling" tests/utils_/test_mem_utils.py tests/utils_/test_gc_utils.py
     tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_worker.py
     tests/model_executor/model_loader/test_direct_loader.py tests/model_executor/model_loader/test_direct_loader_cuda.py)
exec 9>>/tmp/spark-lab-gpu.lock; flock 9
dps=$(docker ps --format "{{.ID}} {{.Image}} {{.Names}}" | tr "\n" ";")
if [ -e "$HOME/spark-hold.json" ]; then hold=present; else hold=absent; fi
{ echo "### $label"; echo "host: $(hostname) (sp8)"; echo "start_utc: $(date -u +%FT%TZ)"; echo "docker_ps_before: ${dps:-none}"; echo "hold_file: $hold"; echo "command: flock /tmp/spark-lab-gpu.lock $(printf "%q " "${cmd[@]}")"; } >> $O/commands.txt
if [ $hold = present ] || [ -n "$dps" ]; then printf "skipped\n\n" >> $O/commands.txt; exit 3; fi
"${cmd[@]}" > $O/$label.txt 2>&1; rc=$?
{ echo "end_utc: $(date -u +%FT%TZ)"; echo "exit_code: $rc"; echo "output: $label.txt"; echo; } >> $O/commands.txt
echo "$label rc=$rc"; exit $rc
