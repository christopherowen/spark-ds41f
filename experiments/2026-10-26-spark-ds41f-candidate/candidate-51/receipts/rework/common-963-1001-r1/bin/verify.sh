#!/bin/bash
# Tree identity of each arm, as the container sees it (no GPU, no timing): every tracked
# file of the expected tree, and the change markers the corrected arm imports.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; IMG=$(cat $W/image-id); O=$W/out/pr-common; mkdir -p $O
run() {  # LABEL ARM ARGS...
  label=$1; arm=$2; shift 2
  mounts=(); while read -r f; do mounts+=(-v "$W/ovl/$arm/$f:/opt/spark-ds41f/candidate/vllm/$f:ro"); done < $W/files-$arm.txt
  cmd=(docker run --rm -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm -v "$W/h:/h:ro" -w /opt/spark-ds41f/candidate/vllm "${mounts[@]}" --entrypoint python3 "$IMG" "$@")
  { echo "### $label"; echo "arm: $arm"; echo "start_utc: $(date -u +%FT%TZ)"; echo "command: $(printf '%q ' "${cmd[@]}")"; } >> $O/commands.txt
  "${cmd[@]}" > $O/$label.txt 2>&1; rc=$?
  { echo "end_utc: $(date -u +%FT%TZ)"; echo "exit_code: $rc"; echo "output: $label.txt"; echo; } >> $O/commands.txt
}
run verify-tree-base base /h/verify_tree.py /h/base-14c5a2dc.txt
run verify-tree-c51r1 c51r1 /h/verify_tree.py /h/c51r1-tree.txt
run verify-overlay-c51r1 c51r1 /h/verify_overlay.py
