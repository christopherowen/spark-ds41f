#!/usr/bin/env bash
# Run on each node with the complete serving cluster stopped.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
readarray -t pin < <(python3 - "$root/config/kernel-trial.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); m=p['memory_saver']
print(m['repository']); print(m['revision']); print(m['version']); print(p['candidate'])
PY
)
test -z "$(docker ps -q --filter name=^dsv41-karmic-kraken$)"
source_dir="$HOME/projects/dgx-spark-memory-saver"
if [ ! -e "$source_dir" ]; then git clone "${pin[0]}" "$source_dir"; fi
test "$(git -C "$source_dir" remote get-url origin)" = "${pin[0]}"
test -z "$(git -C "$source_dir" status --porcelain)"
git -C "$source_dir" fetch origin
git -C "$source_dir" checkout --detach "${pin[1]}"
cd "$source_dir"
./scripts/check
# The fleet already uses this enrolled identity for fan-control; preserve it.
sudo mokutil --test-key /root/.local/share/dgx-spark-fan-control/keys/MOK.der || true
sudo python3 - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from signing import check_pair, require_enrolled
from pathlib import Path
p=Path('/root/.local/share/dgx-spark-fan-control/keys')
check_pair(p/'MOK.priv', p/'MOK.der'); require_enrolled(p/'MOK.der')
PY
version=${pin[2]}
kernel=${pin[3]}
destination="/usr/src/dgx-spark-memory-saver-$version"
test ! -e "$destination"
test -z "$(dkms status -m dgx-spark-memory-saver)"
sudo install -d "$destination/scripts" "$destination/patches" "$destination/packaging"
sudo install -m 0644 dkms.conf provenance.json "$destination/"
sudo install -m 0755 scripts/driver-build scripts/refresh-initramfs "$destination/scripts/"
sudo install -m 0644 scripts/driver_build.py "$destination/scripts/"
sudo install -m 0644 patches/0001-pack-user-leaf-tables.patch "$destination/patches/"
sudo install -m 0644 packaging/enable-packing.patch "$destination/packaging/"
sudo dkms add -m dgx-spark-memory-saver -v "$version"
sudo dkms build -m dgx-spark-memory-saver -v "$version" -k "$kernel"
if [ "$(uname -r)" = "$kernel" ] && [ -d /sys/module/nvidia_uvm ]; then
  if sudo fuser /dev/nvidia-uvm; then echo 'Stop GPU clients first' >&2; exit 1; fi
  sudo modprobe -r nvidia_uvm
fi
sudo dkms install -m dgx-spark-memory-saver -v "$version" -k "$kernel"
sudo depmod "$kernel"
sudo update-initramfs -u -k "$kernel"
./scripts/status --json
