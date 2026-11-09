#!/bin/bash
# verify.sh ARM: no-GPU container with the arm's mounts (taken under the GPU lock to keep sp8's
# queue orderly); prints each overlaid file's SHA-256 as the container sees it. The imports need a
# GPU (TileLang), so they are checked by q.sh's import step through run.sh.
B=$HOME/spark-lab/agents/candidate-53-rework-971; IMG=$(cat $B/image-id); arm=$1
mounts=()
case $arm in
  r1|orig) list=$B/changed-files-$arm.txt; src=$B/ovl/$arm;;
  base) list=$B/no1009-files.txt; src=$B/ovl/no1009;;
esac
while read -r f; do mounts+=(-v "$src/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $list
flock /tmp/spark-lab-gpu.lock docker run --rm --name c53rw971-verify-$arm --network none "${mounts[@]}" -v $B/changed-files-orig.txt:/list:ro \
  -e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 -w /opt/spark-ds41f/candidate/vllm --entrypoint bash $IMG -c '
while read -r f; do echo "$(sha256sum < $f | cut -c1-12) $f"; done < /list'
