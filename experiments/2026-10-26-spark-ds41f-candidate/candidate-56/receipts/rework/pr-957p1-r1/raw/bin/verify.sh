#!/bin/bash
# Tree identity of each arm as its container sees it (CPU-only container that hashes
# files; no GPU, no lock). lists/ARM.txt is the expected tree (git ls-tree -r).
W=$HOME/spark-lab/agents/candidate-56-rework-957p1
IMG=$(cat $W/image-id)
mkdir -p $W/out/verify
for arm in base r1 c56r1; do
  img=$IMG; [ $arm = c56r1 ] && img=$(cat $W/image-id-c56)
  mounts=()
  [ $arm != base ] && while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/files-$arm.txt
  list=$([ $arm = base ] && echo c54-5f021d9a || echo $arm)
  { echo "### verify-tree-$arm $(date -u +%FT%TZ) image $img"
    docker run --rm -v $W/h:/h:ro -v $W/lists:/lists:ro "${mounts[@]}" --entrypoint python3 $img /h/verify_tree.py /lists/$list.txt; } > $W/out/verify/verify-tree-$arm.txt 2>&1
  sed -n 2,4p $W/out/verify/verify-tree-$arm.txt
done
