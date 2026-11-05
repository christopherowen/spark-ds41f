#!/bin/bash
# Make 7.0.0-1019-nvidia-64k the persistent GRUB default (run as root, on 64 KiB, after its checks).
set -euo pipefail
target=7.0.0-1019-nvidia-64k
test "$(uname -r)" = "$target"
uuid=$(findmnt -n -o UUID /)
entry="gnulinux-advanced-$uuid>gnulinux-$target-advanced-$uuid"
grep -F "'gnulinux-$target-advanced-$uuid'" /boot/grub/grub.cfg >/dev/null
printf '# 7.0 64 KiB with memory saver is the normal boot (rollback: 6.17.0-1032-nvidia).\nGRUB_DEFAULT="%s"\n' \
  "$entry" > /etc/default/grub.d/zz-spark-kernel-trial.cfg
update-grub >/dev/null 2>&1
grub-script-check /boot/grub/grub.cfg
grep -F "set default=\"$entry\"" /boot/grub/grub.cfg >/dev/null
test -z "$(grub-editenv list | grep -E '^(next_entry|initrdfail|prev_entry)=.' || true)"
echo "$(hostname): default = $target"
