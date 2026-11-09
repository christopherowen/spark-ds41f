# Sourced by the batch scripts. R = spark-ds41f:1107 image ID (its vLLM tree is
# ed1bdb7a, 1107's patch head 2756478b); each arm is a candidate-50 tree whose files
# that differ from 1107's are mounted read-only over the image's vLLM root.
B=$HOME/spark-lab/agents/candidate-50-rework-1009
R=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
HUB="-v $HOME/.cache/huggingface/hub:/hub:ro"
# mounts <tree>: -v flags for every file the tree changes against 1107's patch head
mounts() {
  local tree=$1 m="" f
  for f in $(git -C "$tree" diff --name-only base1107 HEAD); do m="$m -v $tree/$f:$V/$f:ro"; done
  echo "$m"
}
M_C50=$(mounts $B/vllm-c50)       # candidate-50 as built (original 0778)
M_FIX=$(mounts $B/vllm-fix)       # candidate-50 with the corrected 0778
M_NO=$(mounts $B/vllm-no1009)     # candidate-50 without 0778 (0779, 0780 applied)
SH="-v $B/shim:/shim:ro -v $B/out/json:/out"
