#!/usr/bin/env bash
# Exact cached transaction; run only inside the coordinated exclusive window.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
arm=${1:?new or old}
case "$arm" in new|old) ;; *) exit 2;; esac
test -z "$(docker ps -q --filter name=^dsv41-karmic-kraken$)"
ssh -n dgx1 "python3 -c 'import json,pathlib; h=json.loads((pathlib.Path.home()/\"spark3-hold.json\").read_text()); assert \"Codex driver/CUDA\" in h[\"holder\"]'"
packages="$root/.work/driver-cuda-refresh/packages"
python3 - "$packages" "$arm" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);d=json.loads((p/'transaction.json').read_text())
files=[(n,h) for n,h in d['sha256'].items() if n.startswith(sys.argv[2]+'/')]
assert len(files)==len(d[sys.argv[2]])
for name,h in files: assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
PY
# Remove the separately qualified RM experiment before restoring vendor packages.
if [ "$arm" = old ]; then
  override=/lib/modules/7.0.0-1019-nvidia-64k/updates/driver-cuda-refresh/nvidia.ko
  if [ -f "$override" ]; then
    sudo rm -- "$override"
    sudo depmod 7.0.0-1019-nvidia-64k
    sudo update-initramfs -u -k 7.0.0-1019-nvidia-64k
  fi
fi
# Remove the old DKMS override before replacing its packaged stock provider.
if dkms status -m dgx-spark-memory-saver | grep -q .; then
  for version in 0.2.0 0.3.0; do
    if dkms status -m dgx-spark-memory-saver -v "$version" 2>/dev/null | grep -q .; then
      sudo dkms remove -m dgx-spark-memory-saver -v "$version" --all
    fi
  done
fi
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$packages/$arm/"*.deb
if [ "$arm" = new ]; then
  revision=a7905cb
  version=0.3.0
else
  revision=2dd6ef107616ef13f1c7f87817c07e5aabc9199e
  version=0.2.0
fi
repo="$HOME/projects/dgx-spark-memory-saver"
test -z "$(git -C "$repo" status --porcelain)"
git -C "$repo" fetch origin
git -C "$repo" checkout --detach "$revision"
dest="/usr/src/dgx-spark-memory-saver-$version"
if [ ! -d "$dest" ]; then
  sudo mkdir "$dest"
  git -C "$repo" archive "$revision" | sudo tar -x -C "$dest"
fi
sudo dkms add -m dgx-spark-memory-saver -v "$version"
sudo dkms build -m dgx-spark-memory-saver -v "$version" -k 7.0.0-1019-nvidia-64k
sudo dkms install -m dgx-spark-memory-saver -v "$version" -k 7.0.0-1019-nvidia-64k
sudo depmod 7.0.0-1019-nvidia-64k
sudo update-initramfs -u -k 7.0.0-1019-nvidia-64k
modinfo -k 7.0.0-1019-nvidia-64k nvidia_uvm | head -16
# Reboot is coordinated separately, only after every node's preparation succeeds.
