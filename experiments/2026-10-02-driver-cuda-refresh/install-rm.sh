#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
artifact=${1:?path to signed qualification nvidia.ko}
test -z "$(docker ps -q --filter name=^dsv41-karmic-kraken$)"
test "$(modinfo -F version "$artifact")" = 610.57.04
test "$(modinfo -F vermagic "$artifact" | cut -d' ' -f1)" = 7.0.0-1019-nvidia-64k
test "$(modinfo -F signer "$artifact")" = 'DGX Fleet Secure Boot Module Signature key'
python3 - "$artifact" "$root/experiments/2026-10-02-driver-cuda-refresh/rm-build.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);m=json.load(open(sys.argv[2]));assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
PY
dest=/lib/modules/7.0.0-1019-nvidia-64k/updates/driver-cuda-refresh/nvidia.ko
test ! -e "$dest"
sudo install -D -m 0644 "$artifact" "$dest"
sudo depmod 7.0.0-1019-nvidia-64k
sudo update-initramfs -u -k 7.0.0-1019-nvidia-64k
test "$(modinfo -F filename nvidia)" = "$dest"
modinfo -F srcversion nvidia
