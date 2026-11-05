#!/bin/bash
set -eu
cd /home/swank/spark-lab/collective-pdl
for phase in staged arrived; do
 mkdir -p "compiled-$phase"
 docker run --rm --runtime=runc --name=collective-pdl-cpu-compile --network=none --cpus=2 --memory=8g --memory-swap=8g --pids-limit=256 \
 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e MAX_JOBS=2 -e OMP_NUM_THREADS=2 -e TVM_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONPATH=/source \
 --volume "/home/swank/spark-lab/collective-pdl/sparknet-$phase:/source:ro" \
 --volume /home/swank/spark-lab/collective-pdl/deployment/experiments/2026-11-05-collective-pdl:/check:ro \
 --volume "/home/swank/spark-lab/collective-pdl/compiled-$phase:/out" \
 --entrypoint=python3 sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b /check/compile_sources.py --phase "$phase"
done
