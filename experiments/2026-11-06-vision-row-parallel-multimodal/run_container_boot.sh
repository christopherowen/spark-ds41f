#!/usr/bin/env bash
set -euo pipefail
BOOT="${1:?boot number required}"
cd /tmp
VLLM_PKG="$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)")"
cp /workspace/vllm/vllm/models/deepseek_v4_1/nvidia/tilelang_vision.py "$VLLM_PKG/models/deepseek_v4_1/nvidia/tilelang_vision.py"
cp /workspace/vllm/vllm/models/deepseek_v4_1/nvidia/vl_model.py "$VLLM_PKG/models/deepseek_v4_1/nvidia/vl_model.py"
cp /workspace/vllm/vllm/models/deepseek_v4_1/tilelang/gather.py "$VLLM_PKG/models/deepseek_v4_1/tilelang/gather.py"
cp /workspace/vllm/vllm/multimodal/media/image.py "$VLLM_PKG/multimodal/media/image.py"

python3 - "$VLLM_PKG" << 'PY'
import sys, pathlib
vllm_pkg = pathlib.Path(sys.argv[1])
ws_vllm = pathlib.Path("/workspace/vllm/vllm")
for so in vllm_pkg.rglob("*.so"):
    rel = so.relative_to(vllm_pkg)
    dst = ws_vllm / rel
    if not dst.exists() and not dst.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(so)
ver = vllm_pkg / "_version.py"
dst_ver = ws_vllm / "_version.py"
if ver.exists() and not dst_ver.exists() and not dst_ver.is_symlink():
    dst_ver.symlink_to(ver)
PY

cd /workspace/vllm
pytest --noconftest -v -o addopts="" tests/multimodal/media/test_image.py tests/kernels/test_deepseek_v41_tilelang_vision.py | tee "/workspace/pytest_boot${BOOT}.log"
PYTHONPATH=/workspace/vllm python3 /workspace/bench_vision_rework.py --boot "$BOOT" --out "/workspace/bench_boot${BOOT}.json" | tee "/workspace/bench_boot${BOOT}.log"
