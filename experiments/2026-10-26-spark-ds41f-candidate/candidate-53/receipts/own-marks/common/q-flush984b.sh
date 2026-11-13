#!/bin/bash
B=$HOME/spark-lab/agents/candidate-53-own-marks; cd $B
for k in 1 2; do
  for i in $(seq 1 30); do [ -e $HOME/spark-hold.json ] && exit 9; common/run.sh 984 flushb-c53-boot$k c53 -v $B/harness:/h:ro -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 -e MALLOC_ARENA_MAX=2 -e MALLOC_MMAP_THRESHOLD_=1048576 -- python3 /h/flush984b.py c53-boot$k /out/flushb-c53-boot$k.json; rc=$?; [ $rc -ne 3 ] && break; sleep 60; done
done
echo "FLUSHB DONE $(date -u +%FT%TZ)"
