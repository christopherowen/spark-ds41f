#!/bin/bash
# Tree identity of each arm, as its container sees it (no GPU, no lock: a CPU-only
# container that hashes files). lists/ARM.txt is the expected tree (git ls-tree -r).
B=$HOME/spark-lab/agents/round3-own-marks
IMG=$(cat $B/image-id)
mkdir -p $B/out/common
for arm in base c56p1 c57p2 c57p4 c55dr; do
  mounts=()
  if [ $arm != base ]; then
    while read -r f; do mounts+=(-v "$B/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $B/files-$arm.txt
  fi
  list=$([ $arm = base ] && echo c54-5f021d9a || echo $arm)
  { echo "### verify-tree-$arm $(date -u +%FT%TZ) image $IMG"; 
    docker run --rm -v $B/h:/h:ro -v $B/lists:/lists:ro "${mounts[@]}" --entrypoint python3 $IMG /h/verify_tree.py /lists/$list.txt; } > $B/out/common/verify-tree-$arm.txt 2>&1
  head -4 $B/out/common/verify-tree-$arm.txt | tail -2
done
