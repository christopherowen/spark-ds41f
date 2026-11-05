#!/bin/bash
# Checks on one node after a boot. Run as admin (uses sudo -n). Read-only except a 25 s
# CUDA load in a throwaway container. Order matters: UVM's packing parameter exists only once a CUDA client loaded it.
export LC_ALL=C
echo "== $(hostname) $(date -u +%FT%TZ) up since $(uptime -s)"
echo "kernel $(uname -r) pagesize $(getconf PAGESIZE) driver $(cat /sys/module/nvidia/version 2>/dev/null)"
c=$(cat /proc/cmdline)
echo "cmdline kho=off:$([[ " $c " == *' kho=off '* ]] && echo yes || echo NO) splash:$([[ " $c " == *' splash '* ]] && echo YES || echo no)"
echo "secure boot: $(mokutil --sb-state 2>&1 | head -1); MokNew: $(sudo -n mokutil --list-new 2>&1 | wc -l) lines"
echo "failed units: $(systemctl --failed --no-legend --plain | awk '{print $1}' | tr '\n' ' ')"
echo "default target $(systemctl get-default); dashboard $(systemctl is-enabled dgx-dashboard.service 2>&1)"
echo "cpu governor: $(systemctl show nv-cpu-governor.service -p Result -p ExecMainStatus --value | tr '\n' ' ')" \
  "$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
echo "kernel-memory: $(systemctl show spark-ds41f-kernel-memory.service -p Result -p ExecMainStatus --value | tr '\n' ' ')"
echo "swap: $(swapon --show=NAME,SIZE --noheadings | tr -s ' ' | tr '\n' ' ')"
echo "thp $(cat /sys/kernel/mm/transparent_hugepage/enabled); min_free_kbytes $(sysctl -n vm.min_free_kbytes);" \
  "watermark_boost $(sysctl -n vm.watermark_boost_factor)"
echo "nvme apst $(cat /sys/module/nvme_core/parameters/apst_primary_timeout_ms)/$(cat /sys/module/nvme_core/parameters/apst_secondary_timeout_ms);" \
  "coalescing unit $(systemctl is-enabled nvidia-nvme-interrupt-coalescing.service 2>&1)"
echo "default-route address: $(ip -4 -br addr show "$(ip route show default | awk '{print $5; exit}')" | awk '{print $1, $3}')"
for d in /sys/class/infiniband/*/ports/1; do
  printf '%s %s; ' "$(basename "$(dirname "$(dirname "$d")")")" "$(cut -d' ' -f2 "$d/state")"
done; echo
echo "LTR: $(sudo -n setpci -s 0000:00:00.0 CAP_EXP+28.w) $(sudo -n setpci -s 0002:00:00.0 CAP_EXP+28.w)"
echo "xid this boot: $(sudo -n journalctl -k -b --no-pager | grep -c 'NVRM: Xid')"
echo "kernel warnings this boot: $(sudo -n journalctl -k -b -p warning --no-pager | wc -l)"
if findmnt -n /home/admin/ds41f-1017/blobs >/dev/null; then
  echo "blobs: $(timeout 10 ls /home/admin/ds41f-1017/blobs | wc -l) entries"
else
  echo "blobs: not mounted"
fi
echo "idle: $(nvidia-smi --query-gpu=clocks.sm,power.draw,pstate --format=csv,noheader)"
docker rm -f clockload >/dev/null 2>&1
docker run -d --rm --name clockload --gpus all --entrypoint /usr/bin/python3 spark-ds41f:1030 -c "
import torch,time
a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); b=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16)
torch.cuda.synchronize(); t=time.time(); n=0
while time.time()-t<25:
    c=a@b; n+=1
torch.cuda.synchronize(); print(n)
" >/dev/null
sleep 12
for i in 1 2 3; do
  echo "load: $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu,power.draw,pstate,clocks_event_reasons.active --format=csv,noheader)"
  sleep 3
done
docker wait clockload >/dev/null 2>&1
echo "uvm loaded: $([ -d /sys/module/nvidia_uvm ] && echo yes || echo no);" \
  "pack param: $(cat /sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables 2>&1)"
echo "uvm srcversion loaded $(cat /sys/module/nvidia_uvm/srcversion 2>&1) on-disk $(modinfo -F srcversion nvidia_uvm)" \
  "file $(modinfo -F filename nvidia_uvm)"
echo "MemAvailable $(awk '/MemAvailable/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
echo "CHECK_END"
