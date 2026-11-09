B=$HOME/spark-lab/agents/candidate-49-rework-996-r2
C=sha256:9f51f2e2c5ab728e96490d794bff7b83a147d6d3d61333c5b60b2cff662049cd   # spark-ds41f:candidate-49
R=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce   # spark-ds41f:1107
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
# the r2 stack: 1107 image with 0762..0766 and the r2 0767 mounted over it
M=""; for f in $(cd $B/vllm-stack && { git diff --name-only base HEAD; git diff --name-only HEAD; } | sort -u); do M="$M -v $B/vllm-stack/$f:$V/$f:ro"; done
H="-v $B/shim/bench_996_route_local.py:/tmp/bench_996_route_local.py:ro -v $B/shim/exact_996.py:/tmp/exact_996.py:ro -v $B/shim/cpu_996.py:/tmp/cpu_996.py:ro -v $B/out/json:/out"
NEW_TESTS='inference_mode or last_one_forward or outside_a_forward'
# round 1 control: round 1 tree (rework-996 873df17) mounted the same way
M_R1=""; for f in $(cd $B/vllm-stack && git diff --name-only base HEAD); do M_R1="$M_R1 -v $B/vllm-r1ctl/$f:$V/$f:ro"; done
# round 1 test files, unchanged, over round 2 router
M_X=""; for f in $(cd $B/vllm-stack && git diff --name-only base HEAD); do s=$B/vllm-r1ctl/$f; [ "$f" = vllm/models/deepseek_v4_1/tilelang/router.py ] && s=$B/vllm-stack/$f; M_X="$M_X -v $s:$V/$f:ro"; done
