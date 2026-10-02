#!/bin/bash
# Host preparation only. Never reboot, load a module, or restart a service.
set -euo pipefail
export LC_ALL=C
target=7.0.0-1019-nvidia-64k
fallback=7.0.0-1019-nvidia
version=7.0.0-1019.19~24.04.2
container=dsv41-karmic-kraken
backup=/var/lib/spark3/kernel-64k-preparation
test "$(id -u)" = 0
test "$(uname -r)" = "$fallback"
test "$(cat /sys/module/nvidia/version)" = 580.178.04
test -z "$(grub-editenv list | grep -E '^(next_entry|initrdfail|prev_entry)=.' || true)"
before=$(docker inspect -f '{{.Id}} {{.State.StartedAt}} {{.State.Running}}' "$container")
[[ "$before" = *' true' ]]
mkdir -p "$backup"
test ! -e "$backup/grub.cfg.before" || { echo 'Existing preparation receipt; inspect before rerunning'; exit 1; }
cp -a /boot/grub/grub.cfg "$backup/grub.cfg.before"
cp -a /etc/default/grub "$backup/grub.default.before"
cp -a /etc/default/grub.d "$backup/grub.d.before"
grub-editenv list > "$backup/grubenv.before"
printf '%s\n' "$before" > "$backup/container.before"
uuid=$(findmnt -n -o UUID /)
test -n "$uuid"
entry="gnulinux-advanced-$uuid>gnulinux-$fallback-advanced-$uuid"
grep -F "'gnulinux-$fallback-advanced-$uuid'" /boot/grub/grub.cfg >/dev/null
pin=/etc/default/grub.d/zz-spark-kernel-trial.cfg
test ! -e "$pin"
printf '# Keep the existing kernel as normal boot while 64 KiB is staged.\nGRUB_DEFAULT="%s"\n' "$entry" > "$pin"
update-grub
grep -F "set default=\"$entry\"" /boot/grub/grub.cfg >/dev/null
# Listing mode prevents needrestart from restarting services after installation.
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get -y --no-remove --no-install-recommends install \
  "linux-image-$target=$version" \
  "linux-modules-$target=$version" \
  "linux-headers-$target=$version" \
  "linux-tools-$target=$version" \
  "linux-modules-nvidia-580-open-$target=$version+1"
dkms autoinstall -k "$target"
depmod "$target"
update-initramfs -u -k "$target"
update-grub
grub-script-check /boot/grub/grub.cfg
grep -F "set default=\"$entry\"" /boot/grub/grub.cfg >/dev/null
grep -F "'gnulinux-$target-advanced-$uuid'" /boot/grub/grub.cfg >/dev/null
grep -qx 'CONFIG_ARM64_64K_PAGES=y' "/boot/config-$target"
test -s "/boot/vmlinuz-$target"
lsinitramfs "/boot/initrd.img-$target" > "$backup/initrd.contents"
grep -F "lib/modules/$target" "$backup/initrd.contents" >/dev/null
for module in nvidia nvidia_uvm nvidia_drm mlx5_core mlx5_ib dgx_ec_fan_control; do
  test "$(modinfo -k "$target" -F vermagic "$module" | cut -d' ' -f1)" = "$target"
done
test "$(modinfo -k "$target" -F version nvidia)" = 580.178.04
dkms status -m dgx-spark-fan-control -k "$target" | grep -F ': installed'
test "$(uname -r)" = "$fallback"
test "$(getconf PAGESIZE)" = 4096
after=$(docker inspect -f '{{.Id}} {{.State.StartedAt}} {{.State.Running}}' "$container")
test "$after" = "$before"
printf '%s\n' "$after" > "$backup/container.after"
dpkg-query -W "linux-image-$target" "linux-modules-$target" "linux-headers-$target" \
  "linux-modules-nvidia-580-open-$target" > "$backup/packages.after"
sha256sum "/boot/vmlinuz-$target" "/boot/config-$target" > "$backup/kernel.sha256"
date -u +%FT%TZ > "$backup/completed"
echo 'PREPARED: candidate installed, old kernel remains default, serving unchanged; no trial boot performed.'
