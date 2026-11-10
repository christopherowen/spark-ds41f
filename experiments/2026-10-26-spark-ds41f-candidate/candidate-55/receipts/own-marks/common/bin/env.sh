# Recipe environment for the one-GPU runs (releases/spark-ds41f-1107/cluster.json "environment",
# the entries that act in one process), as docker -e options.
RECIPE_ENV=(-e CUDA_DEVICE_MAX_CONNECTIONS=32 -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
  -e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 -e SPARK_DS41F_DSPARK_MAIN_PROJ_TP=1 -e SPARK_DS41F_DSPARK_VOCAB_PARALLEL=1
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_DS41_HEAD_DTYPE=float32 -e VLLM_DS41_PACKED_BF16_LM_HEAD=1
  -e VLLM_L2_PREFETCH_GRID=2 -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_LATE_ADMISSION_LAYER=30
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_DS41_KERNEL_BACKEND=tilelang
  -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 -e MALLOC_ARENA_MAX=2 -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_PLUGINS= -e TRITON_CACHE_DIR=/root/.tilelang/triton -e VLLM_CACHE_ROOT=/root/.tilelang/vllm)
HUB=(-v $HOME/.cache/huggingface/hub:/hf/hub:ro)
NSYS=(-v /opt/nvidia/nsight-systems/2025.3.2:/opt/nsys:ro)
