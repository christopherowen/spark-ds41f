#!/usr/bin/env bash
# Build the pinned one-sided GPUNetIO samples in an isolated, disposable tree.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/.work/gpunetio-586453728bca"
revision=586453728bcab2d4c50574924dc6cf43543c9ed4
if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/NVIDIA-DOCA/gpunetio.git "$source_dir"
fi
[[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]]
git -C "$source_dir" checkout --detach "$revision"
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]]
make -C "$source_dir" -j4 CUDA_ARCH=121 CUDA_HOME=/usr/local/cuda-13.0
