#!/bin/bash
# Run from this published checkout before the first trial boot. No reboot here.
set -euo pipefail
test "$(id -u)" = 0
root=$(cd "$(dirname "$0")/../.." && pwd)
backup=/var/lib/spark3/kernel-64k-boot
mkdir -p "$backup"
test -e "$backup/fstab.before" || cp -a /etc/fstab "$backup/fstab.before"
if [ ! -e /swap-64k.img ]; then
  (umask 077; fallocate -l 16G /swap-64k.img)
  chmod 600 /swap-64k.img
  mkswap --pagesize 65536 /swap-64k.img
fi
python3 - <<'PY'
from pathlib import Path
with open('/swap-64k.img', 'rb') as f:
    f.seek(65536 - 10)
    assert f.read(10) == b'SWAPSPACE2', '64 KiB swap signature missing'
p = Path('/etc/fstab')
lines = p.read_text().splitlines()
found = 0
for i, line in enumerate(lines):
    words = line.split()
    if words and words[0] == '/swap.img':
        assert len(words) >= 6 and words[2] == 'swap'
        words[3] = ','.join([x for x in words[3].split(',') if x not in ('sw', 'auto', 'noauto')] + ['noauto'])
        lines[i] = '\t'.join(words)
        found += 1
assert found == 1, 'expected exactly one existing swap entry'
p.write_text('\n'.join(lines) + '\n')
PY
install -m 0755 "$root/host/kernel-memory/spark3-kernel-memory" /usr/local/sbin/spark-kernel-memory
install -m 0644 "$root/host/kernel-memory/spark3-kernel-memory.service" /etc/systemd/system/spark3-kernel-memory.service
systemctl daemon-reload
systemctl enable spark3-kernel-memory.service
systemd-analyze verify /etc/systemd/system/spark3-kernel-memory.service
echo 'Boot memory support installed; current swap and running memory policy unchanged.'
