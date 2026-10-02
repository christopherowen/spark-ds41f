#!/usr/bin/env bash
# Separate RM experiment; this is deliberately outside the UVM-only saver package.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
source_dir=/usr/src/nvidia-610.57.04
build="$root/.work/driver-cuda-refresh/rm610"
test -z "$(docker ps -q --filter name=^dsv41-karmic-kraken$)"
test "$(dpkg-query -W -f='${Version}' nvidia-kernel-source-610-open)" = 610.57.04-0ubuntu0.24.04.3
echo "50378b35ed54affcdb314e77ba9a495801070cbe2877f3b25ba6f7f41660e42e  $source_dir/common/inc/nv-linux.h" | sha256sum -c -
test ! -e "$build"
cp -a "$source_dir" "$build"
patch -d "$build" -p1 --fuzz=0 --batch -i "$root/experiments/2026-10-02-driver-cuda-refresh/0001-rm-arm64-dma-alignment.patch"
make -C "$build" -j4 CC=gcc-13 NV_KERNEL_MODULES=nvidia KERNEL_UNAME=7.0.0-1019-nvidia-64k SYSSRC=/lib/modules/7.0.0-1019-nvidia-64k/build modules
sudo /lib/modules/7.0.0-1019-nvidia-64k/build/scripts/sign-file sha512 /root/.local/share/dgx-spark-fan-control/keys/MOK.priv /root/.local/share/dgx-spark-fan-control/keys/MOK.der "$build/nvidia.ko"
modinfo "$build/nvidia.ko"
sha256sum "$build/nvidia.ko"
