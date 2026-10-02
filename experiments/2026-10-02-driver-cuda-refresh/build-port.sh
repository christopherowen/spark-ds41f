#!/usr/bin/env bash
# Compile against extracted, hash-checked R610 source before changing packages.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
test -z "$(docker ps -q --filter name=^dsv41-karmic-kraken$)"
work="$root/.work/driver-cuda-refresh"
repo="$HOME/projects/dgx-spark-memory-saver"
git -C "$repo" fetch origin
pin=a7905cb
export_dir="$work/memory-saver-610-source"
mkdir -p "$export_dir"
git -C "$repo" archive "$pin" | tar -x -C "$export_dir"
source_dir="$work/source610/usr/src/nvidia-610.57.04"
python3 - "$export_dir/provenance.json" "$source_dir" <<'PY'
import json,hashlib,pathlib,sys
m=json.load(open(sys.argv[1]));r=pathlib.Path(sys.argv[2])
for name,h in m['upstream']['file_sha256'].items(): assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h
PY
build="$work/build610"
test ! -e "$build"
cp -a "$source_dir" "$build"
patch -d "$build" -p1 --fuzz=0 --batch -i "$export_dir/patches/0002-pack-user-leaf-tables-610.patch"
patch -d "$build" -p1 --fuzz=0 --batch -i "$export_dir/packaging/enable-packing.patch"
make -C "$build" -j4 CC=gcc-13 NV_KERNEL_MODULES='nvidia nvidia-uvm' KERNEL_UNAME=7.0.0-1019-nvidia-64k SYSSRC=/lib/modules/7.0.0-1019-nvidia-64k/build modules
modinfo "$build/nvidia-uvm.ko"
sha256sum "$build/nvidia-uvm.ko"
