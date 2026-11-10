# Sourced by the batch scripts. R = spark-ds41f:candidate-50 image ID (its vLLM tree is
# d5778ab5, candidate-50 patch head ba7400d9). Each arm is a full tree whose files that
# differ from ba7400d9 are mounted read-only over the image vLLM root:
#   base = candidate-54 + 957-p1 (patch head 0c4142a2, tree 3959b211)
#   cand = base + the rebased 956 (patch head as committed in $B/vllm r956)
B=$HOME/spark-lab/agents/round3-956
R=sha256:8c1ea1f62f62c3e2adc03b645c091a5499cf7163101aca32875c894b96f155db
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
IMG_BASE=ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9
HUB="-v $HOME/.cache/huggingface/hub:/hub:ro"
mounts() {
  local tree=$1 m="" f
  for f in $(git -C "$tree" diff --name-only $IMG_BASE HEAD); do m="$m -v $tree/$f:$V/$f:ro"; done
  echo "$m"
}
M_BASE=$(mounts $B/tree-base)
M_CAND=$(mounts $B/tree-cand)
SH="-v $B/shim:/shim:ro -v $B/out/json:/out"
TL="-v $B/tlcache:/root/.tilelang/cache"
# The serving environment entries that select code paths (config/cluster.json):
# no breakable graphs (so the fused SP front and the decode front run), CUDA connections.
ENVS="-e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_V2_MODEL_RUNNER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
ENVS_BREAKABLE="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
