#!/bin/bash
# Prepare one node, booted on 6.17.0-1032-nvidia (4 KiB), for 7.0.0-1019-nvidia-64k + memory saver 0.4.1, as this
# site ran it. Run as root with the engine stopped. No reboot and no module load; 6.17 stays the boot default.
# Safe to re-run: every step checks its own state. Assumes: the packages are already in the apt cache (downloaded
# once, copied to each node), a MOK key pair at $keys enrolled at each console, the memory saver checked out at
# $msrc, and this repository at $repo. Site values (paths, the toolkit version) are this site's.
set -euo pipefail
export LC_ALL=C
stock=6.17.0-1032-nvidia
target=7.0.0-1019-nvidia-64k
kv=7.0.0-1019.19~24.04.2
drv=580.178.04
src_ver=580.178.04-0ubuntu0.24.04.1
msv=0.4.1
mrev=ec1b17a8ac713c6357b8f22983ded8503a524849
signer="DGX Spark local module signing"
keys=/root/.local/share/dgx-spark-memory-saver/keys
msrc=/home/admin/ds41f/dgx-spark-memory-saver
repo=/home/admin/projects/spark-ds41f
backup=/var/lib/site-kernel-64k
step() { echo "== $(date -u +%FT%TZ) $(hostname) $*"; }

step preflight
test "$(id -u)" = 0
test "$(uname -r)" = "$stock"
test "$(getconf PAGESIZE)" = 4096
test -z "$(docker ps -q --filter name='^spark-ds41f$')"
test -z "$(grub-editenv list | grep -E '^(next_entry|initrdfail|prev_entry)=.' || true)"
[[ "$(mokutil --test-key "$keys/MOK.der" 2>&1)" == *'already enrolled'* ]]
test "$(stat -c %a "$keys/MOK.priv")" = 600
# As admin: a system unit has no SUDO_UID, so root's git refuses admin-owned repos.
test "$(runuser -u admin -- git -C "$msrc" rev-parse HEAD)" = "$mrev"
test -z "$(runuser -u admin -- git -C "$msrc" status --porcelain)"
test -f "$repo/experiments/2026-10-02-kernel-64k/install-boot-support.sh"
docker_since=$(systemctl show docker -p ActiveEnterTimestamp --value)

step backup
mkdir -p "$backup"
if [ ! -e "$backup/grub.cfg.before" ]; then
  cp -a /etc/default/grub "$backup/grub.default.before"
  cp -a /etc/default/grub.d "$backup/grub.d.before"
  cp -a /etc/fstab "$backup/fstab.before"
  [ -d /etc/dkms ] && cp -a /etc/dkms "$backup/dkms.before"
  grub-editenv list > "$backup/grubenv.before"
  systemctl get-default > "$backup/default-target.before"
  sysctl vm.watermark_boost_factor vm.min_free_kbytes > "$backup/sysctl.before"
  cat /sys/module/nvme_core/parameters/apst_primary_timeout_ms \
    /sys/module/nvme_core/parameters/apst_secondary_timeout_ms > "$backup/nvme-apst.before"
  cat /proc/cmdline > "$backup/cmdline.before"
  dpkg-query -W > "$backup/packages.before"
  cp -a /boot/grub/grub.cfg "$backup/grub.cfg.before"
fi

step "grub pin to $stock, splash off"
uuid=$(findmnt -n -o UUID /)
test -n "$uuid"
stock_entry="gnulinux-advanced-$uuid>gnulinux-$stock-advanced-$uuid"
grep -F "'gnulinux-$stock-advanced-$uuid'" /boot/grub/grub.cfg >/dev/null
printf '# Keep %s as the normal boot while 7.0 64 KiB is staged.\nGRUB_DEFAULT="%s"\n' \
  "$stock" "$stock_entry" > /etc/default/grub.d/zz-spark-kernel-trial.cfg
# Upstream's host/recovery/grub-console.cfg: drop "splash", keep "quiet".
install -m 0644 "$repo/host/recovery/grub-console.cfg" /etc/default/grub.d/zz-spark-console.cfg
update-grub
grub-script-check /boot/grub/grub.cfg
grep -F "set default=\"$stock_entry\"" /boot/grub/grub.cfg >/dev/null

step "apt install (staged packages only, no downloads)"
pk=(nvidia-spark-grub-kho
  "linux-image-$target=$kv" "linux-modules-$target=$kv" "linux-headers-$target=$kv" "linux-tools-$target=$kv"
  "linux-modules-nvidia-580-open-$target" nvidia-driver-580-open "linux-modules-nvidia-580-open-$stock" dkms)
# The only removal allowed: nodes that still carry the 580.173 module for the older 6.17.0-1029 kernel must drop
# it, because it depends on the old driver exactly. 6.17.0-1032 stays the rollback kernel.
removals=$(apt-get -s install "${pk[@]}" | sed -n 's/^Remv \([^ ]*\) .*/\1/p')
for r in $removals; do
  [ "$r" = linux-modules-nvidia-580-open-6.17.0-1029-nvidia ] || { echo "unexpected removal: $r"; exit 1; }
done
echo "removals: ${removals:-none}"
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get -y -q --no-download \
  -o Dpkg::Options::=--force-confold install "${pk[@]}"

step "verify packages and the 6.17 rollback"
grep -qx 'CONFIG_ARM64_64K_PAGES=y' "/boot/config-$target"
test -s "/boot/vmlinuz-$target"
test "$(modinfo -k "$stock" -F version nvidia)" = "$drv"
test "$(modinfo -k "$target" -F version nvidia)" = "$drv"
for m in nvidia nvidia_uvm nvidia_drm mlx5_core mlx5_ib; do
  test "$(modinfo -k "$target" -F vermagic "$m" | cut -d' ' -f1)" = "$target"
done
test "$(dpkg-query -W -f='${Version}' nvidia-kernel-source-580-open)" = "$src_ver"
test "$(dpkg-query -W -f='${Version}' nvidia-container-toolkit)" = 1.20.0-1
test "$(systemctl show docker -p ActiveEnterTimestamp --value)" = "$docker_since"

step "dkms signing key"
grep -H -E '^[[:space:]]*(mok_signing_key|mok_certificate)=' /etc/dkms/framework.conf /etc/dkms/framework.conf.d/*.conf \
  2>/dev/null | grep -v "$keys" && { echo 'another DKMS signing identity is configured; inspect it'; exit 1; }
mkdir -p /etc/dkms/framework.conf.d
printf 'mok_signing_key="%s/MOK.priv"\nmok_certificate="%s/MOK.der"\n' "$keys" "$keys" \
  > /etc/dkms/framework.conf.d/memory-saver-mok.conf

step "memory saver $msv for $target"
runuser -u admin -- bash -c "cd '$msrc' && ./scripts/check"
dest=/usr/src/dgx-spark-memory-saver-$msv
if [ -z "$(dkms status -m dgx-spark-memory-saver -v "$msv")" ]; then
  test ! -e "$dest"
  install -d "$dest/scripts" "$dest/patches" "$dest/packaging"
  (
    cd "$msrc"
    install -m 0644 dkms.conf compatibility.json provenance.json provenance-610.json "$dest/"
    install -m 0755 scripts/driver-build scripts/refresh-initramfs "$dest/scripts/"
    install -m 0644 scripts/driver_build.py "$dest/scripts/"
    install -m 0644 patches/0001-pack-user-leaf-tables.patch patches/0002-pack-user-leaf-tables-610.patch "$dest/patches/"
    install -m 0644 packaging/enable-packing.patch "$dest/packaging/"
  )
  dkms add -m dgx-spark-memory-saver -v "$msv"
fi
state=$(dkms status -m dgx-spark-memory-saver -v "$msv" -k "$target")
case "$state" in
  *installed*) ;;
  *built*) dkms install -m dgx-spark-memory-saver -v "$msv" -k "$target" ;;
  *) dkms build -m dgx-spark-memory-saver -v "$msv" -k "$target"
     dkms install -m dgx-spark-memory-saver -v "$msv" -k "$target" ;;
esac
dkms status
[[ "$(modinfo -k "$target" -F filename nvidia_uvm)" == */updates/dkms/* ]]
test "$(modinfo -k "$target" -F version nvidia_uvm)" = "$drv"
test "$(modinfo -k "$target" -F signer nvidia_uvm)" = "$signer"
params=$(modinfo -k "$target" -p nvidia_uvm)
[[ "$params" == *uvm_pack_sysmem_leaf_tables* ]]
# 6.17 keeps the packaged UVM.
[[ "$(modinfo -k "$stock" -F filename nvidia_uvm)" != */updates/dkms/* ]]

step "boot support: /swap-64k.img and spark-ds41f-kernel-memory.service"
bash "$repo/experiments/2026-10-02-kernel-64k/install-boot-support.sh"

step "headless baseline"
systemctl disable --now dgx-dashboard.service dgx-dashboard-admin.service
systemctl set-default multi-user.target
printf '# Count MemAvailable without the boost reserve (as in the upstream receipts).\nvm.watermark_boost_factor = 0\n' \
  > /etc/sysctl.d/90-watermark-boost.conf
sysctl -q -p /etc/sysctl.d/90-watermark-boost.conf
systemctl mask --now nvidia-nvme-interrupt-coalescing.service
/usr/bin/nvidia-nvme-interrupt-coalescing.sh disable
printf 'options nvme_core apst_primary_timeout_ms=2000 apst_secondary_timeout_ms=60000\n' > /etc/modprobe.d/nvme-apst.conf
echo 2000 > /sys/module/nvme_core/parameters/apst_primary_timeout_ms
echo 60000 > /sys/module/nvme_core/parameters/apst_secondary_timeout_ms
for f in /sys/class/nvme/nvme*/power/pm_qos_latency_tolerance_us; do
  [ -e "$f" ] || continue
  v=$(cat "$f"); echo "$v" > "$f"
done

step "initramfs and grub, final checks"
depmod "$target"
update-initramfs -u -k "$target"
update-initramfs -u -k "$stock"
update-grub
grub-script-check /boot/grub/grub.cfg
grep -F "set default=\"$stock_entry\"" /boot/grub/grub.cfg >/dev/null
grep -F "'gnulinux-$target-advanced-$uuid'" /boot/grub/grub.cfg >/dev/null
grep -m1 -E "^[[:space:]]*linux[[:space:]]+/boot/vmlinuz-$target " /boot/grub/grub.cfg | tee "$backup/target-cmdline"
grep -qw 'kho=off' "$backup/target-cmdline"
if grep -qw splash "$backup/target-cmdline"; then echo 'splash still on the 64 KiB command line'; exit 1; fi
lsinitramfs "/boot/initrd.img-$target" > "$backup/initrd.contents"
grep -F "lib/modules/$target" "$backup/initrd.contents" >/dev/null
test -z "$(grub-editenv list | grep -E '^(next_entry|initrdfail|prev_entry)=.' || true)"
# A key queued by apt or DKMS would put MokManager on the next, headless, boot.
newkeys=$(mokutil --list-new 2>&1 || true)
[[ -z "$newkeys" || "$newkeys" == *'is empty'* ]]
test "$(uname -r)" = "$stock"
test "$(systemctl show docker -p ActiveEnterTimestamp --value)" = "$docker_since"
dpkg-query -W > "$backup/packages.after"
date -u +%FT%TZ > "$backup/completed"
step "PREPARED: $target staged and signed, $stock still the default, nothing rebooted"
