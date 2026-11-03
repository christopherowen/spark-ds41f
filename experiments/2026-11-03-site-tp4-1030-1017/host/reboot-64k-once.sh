#!/bin/bash
# One-shot reboot of one worker into 7.0.0-1019-nvidia-64k, then check-boot.sh. Run on the head:
#   bash reboot-64k-once.sh <worker ssh host>
# 6.17.0-1032 stays the GRUB default until every node has passed.
set -u
h=$1
ssh -o BatchMode=yes "$h" 'u=$(findmnt -n -o UUID /); t="gnulinux-advanced-$u>gnulinux-7.0.0-1019-nvidia-64k-advanced-$u"
  sudo -n grep -qF "gnulinux-7.0.0-1019-nvidia-64k-advanced-$u" /boot/grub/grub.cfg && sudo -n grub-reboot "$t" &&
  sudo -n grub-editenv list | grep -q "^next_entry=$t\$" && echo "next_entry set" && sudo -n systemctl reboot' || exit 1
echo "$(date -u +%T) $h reboot sent"
sleep 10
for i in $(seq 1 120); do
  k=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" uname -r 2>/dev/null)
  [ -n "$k" ] && break
  sleep 10
done
echo "$(date -u +%T) $h up: ${k:-NOT BACK after 20 min}"
[ "$k" = 7.0.0-1019-nvidia-64k ] || exit 1
sleep 20
scp -q "$(dirname "$0")/check-boot.sh" "$h":/tmp/check-boot.sh
ssh -o BatchMode=yes "$h" 'bash /tmp/check-boot.sh; sudo -n grub-editenv list'
