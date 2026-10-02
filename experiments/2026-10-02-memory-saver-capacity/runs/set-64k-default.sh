set -euo pipefail
kernel=7.0.0-1019-nvidia-64k
test "$(uname -r)" = "$kernel"
test "$(getconf PAGESIZE)" = 65536
test "$(cat /sys/module/nvidia_uvm/srcversion)" = 34683B82C2D3339BBD2EEC9
test "$(sudo cat /sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables)" = Y
root_uuid=$(findmnt -n -o UUID /)
test -n "$root_uuid"
entry="gnulinux-advanced-$root_uuid>gnulinux-$kernel-advanced-$root_uuid"
sudo grep -F "'gnulinux-$kernel-advanced-$root_uuid'" /boot/grub/grub.cfg
pin=/etc/default/grub.d/zz-spark-kernel-trial.cfg
backup=/var/lib/spark3/memory-saver-capacity
sudo install -d -m 0755 "$backup"
if [ ! -e "$backup/grub-default-before.cfg" ]; then sudo cp -a "$pin" "$backup/grub-default-before.cfg"; fi
printf '# Validated 64 KiB memory-saver serving default; 4 KiB entry retained.\nGRUB_DEFAULT="%s"\n' "$entry" | sudo tee "$pin"
sudo update-grub
sudo grub-script-check /boot/grub/grub.cfg
sudo grep -F "set default=\"$entry\"" /boot/grub/grub.cfg
sudo grub-editenv list
