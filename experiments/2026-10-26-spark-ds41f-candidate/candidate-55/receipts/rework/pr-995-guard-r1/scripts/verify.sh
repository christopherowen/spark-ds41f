#!/bin/bash
# verify.sh ARM: a no-GPU container with the arm's mounts (taken under the GPU lock, so it never
# overlaps another agent's job): each mounted file's SHA-256 as the container sees it, and
# whether the mounted tilelang_copy.py carries the guard (the import path is recorded by the GPU
# harness: tilelang does not import without a GPU).
W=$HOME/spark-lab/agents/round3-995-guard; IMG=$(cat $W/image-id); arm=$1
mounts=()
while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/files.txt
exec 9>>/tmp/spark-lab-gpu.lock; flock 9
if [ -e "$HOME/spark-hold.json" ] || [ -n "$(docker ps -q)" ]; then echo "skipped: hold file or running container"; exit 3; fi
docker run --rm --network none "${mounts[@]}" -v $W/files.txt:/list:ro -w /opt/spark-ds41f/candidate/vllm --entrypoint bash $IMG -c '
while read -r f; do echo "$(sha256sum < $f | cut -c1-12) $f"; done < /list
F=vllm/v1/worker/gpu/spec_decode/tilelang_copy.py
echo "copy_alloc guard (if slot < 0:): $(grep -c "if slot < 0:" $F)"
echo "copy_apply guard (slot >= 0 and start >= 0): $(grep -c "slot >= 0 and start >= 0" $F)"
echo "cumulative survival (0789): $(grep -c dspark_survival $F)"
echo "two-tier scan (0790): $(grep -c sparse_base $F)"
echo "base decay kept: $(grep -c "accepted >= live, T.min(doublings + 1, MAX_DOUBLINGS), 0" $F); step-down decay: $(grep -c "T.max(doublings - 1, 0)" $F)"'
