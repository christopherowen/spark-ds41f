#!/usr/bin/env bash
# #978's own mark, one session on sp8, run under flock /tmp/spark-lab-gpu.lock:
# the PR's bench_mm_prefix_fastpath.py (67792978) ABBA-style over three arms,
#   base (spark-ds41f:candidate-54 as built), r1 (candidate-55-r1's changed files
#   mounted over it), c55 (candidate-55's changed files mounted over it),
# in the order base r1 c55 c55 r1 base, then the identity check on base and r1.
set -euo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978
IMAGE=spark-ds41f:candidate-54
log=$WD/exp/receipts/own-mark-session.log
mounts() {
  local arm=$1
  [[ -z "$arm" ]] && return 0
  (cd "$WD/$arm" && find . -type f | sed 's|^\./||' | sort) | while IFS= read -r f; do
    printf -- '-v\n%s\n' "$WD/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"
  done
}
run() {  # run <label> <arm-dir or ""> <python args...>
  local label=$1 arm=$2; shift 2
  local -a m=()
  mapfile -t m < <(mounts "$arm")
  echo "$(date -u +%FT%TZ) start $label arm=${arm:-none} mounts=$(( ${#m[@]} / 2 ))" | tee -a "$log"
  local rc=0
  docker run --rm --gpus all --ipc=host "${m[@]}" -v "$WD/exp:/opt/exp" \
    -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 "$IMAGE" "$@" \
    > "$WD/exp/receipts/$label.txt" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) end $label exit=$rc" | tee -a "$log"
}
echo "$(date -u +%FT%TZ) session start, image $(docker image inspect $IMAGE --format '{{.Id}}'), host $(hostname), load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$log"
run bench-base-boot1 ""     /opt/exp/bench_mm_prefix_fastpath.py base /opt/exp/receipts/bench-base-boot1.json
run bench-r1-boot1   arm-r1  /opt/exp/bench_mm_prefix_fastpath.py candidate-55-r1 /opt/exp/receipts/bench-r1-boot1.json
run bench-c55-boot1  arm-c55 /opt/exp/bench_mm_prefix_fastpath.py candidate-55 /opt/exp/receipts/bench-c55-boot1.json
run bench-c55-boot2  arm-c55 /opt/exp/bench_mm_prefix_fastpath.py candidate-55 /opt/exp/receipts/bench-c55-boot2.json
run bench-r1-boot2   arm-r1  /opt/exp/bench_mm_prefix_fastpath.py candidate-55-r1 /opt/exp/receipts/bench-r1-boot2.json
run bench-base-boot2 ""     /opt/exp/bench_mm_prefix_fastpath.py base /opt/exp/receipts/bench-base-boot2.json
run identity-base    ""     /opt/exp/identity_978.py base /opt/exp/receipts/identity-base.json
run identity-r1      arm-r1  /opt/exp/identity_978.py candidate-55-r1 /opt/exp/receipts/identity-r1.json
echo "$(date -u +%FT%TZ) session end, load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$log"
