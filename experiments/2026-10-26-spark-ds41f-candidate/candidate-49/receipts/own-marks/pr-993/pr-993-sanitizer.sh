. $HOME/spark-lab/agents/candidate-49-own-marks/batches/env.sh
# README section 4 names test_indexer_reads_only_visible_rows under compute-sanitizer initcheck; the image has no
# compute-sanitizer, so the scored pytest skipped it. Here the host's CUDA 13.0 compute-sanitizer is mounted read-only.
for boot in 1 2; do
  $B/run.sh 993 sanitizer-candidate49-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name c49om-993-san-b$boot --entrypoint "" -w $V -v /usr/local/cuda-13.0/compute-sanitizer:/usr/local/cuda-13.0/compute-sanitizer:ro -e COMPUTE_SANITIZER=/usr/local/cuda-13.0/compute-sanitizer/compute-sanitizer $C pytest --noconftest -v -rs tests/kernels/attention/test_deepseek_v41_tilelang.py -k test_indexer_reads_only_visible_rows
done
