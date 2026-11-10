# Sourced by the batch scripts (round 3 r4: #956 on candidate-54 without #957 part 1).
# R = spark-ds41f:candidate-56 image ID (built on sp8; its vLLM tree is candidate-54 + 0795 +
# 0796 as first rebased, patch head abb26187). Each arm is a full tree whose files that differ
# from the image's are mounted read-only over the image vLLM root:
#   base = candidate-54 (patch head 4e8133ff): 10 files (0795's and 0796's) at candidate-54's versions
#   cand = candidate-54 + the r4 rebase of 956 (r956r4): 4 files differ from the image
B=$HOME/spark-lab/agents/round3-956-r4
R=sha256:0f65de863418e6ebfc08562b8ac6be43ffea51a0db1e845cafe416cdfbf474bc
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
IMG_BASE=abb261871df2043ec5edee7e5ff2d1c42944b009
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
# The serving environment entries that select code paths (config/cluster.json).
ENVS="-e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_V2_MODEL_RUNNER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
ENVS_BREAKABLE="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
