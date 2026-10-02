# 64 KiB kernel preparation

Owner request: install matching 64 KiB kernels on all three nodes and add
read-only doctor checks. This stages a future trial; it does not authorize a
reboot, service restart, benchmark, or promotion. The promoted container and
`config/cluster.json` are unchanged.

Base: deployment branch `origin/main` at `01ce95d`; r5o production baseline.
Candidate package versions and the retained fallback kernel are in
`config/kernel-trial.json`. The same four exact packages are added to each node,
without upgrades or removals. The running kernel remains `7.0.0-1019-nvidia`
with 4 KiB pages and NVIDIA driver 580.178.04.

## Preparation

`prepare.sh` is the root host operation. It first requires the expected running
kernel/driver, a running production container, and no pending GRUB one-shot or
recovery override. It backs up the boot configuration to
`/var/lib/spark3/kernel-64k-preparation`, then writes
`/etc/default/grub.d/zz-spark-kernel-trial.cfg` with the explicit old-kernel
submenu and entry identifiers. This prevents installation from silently changing
the default kernel when GRUB sorts its entries.

It installs the candidate kernel image, modules, headers and matching NVIDIA
modules, builds the fan-control DKMS module, regenerates initramfs/GRUB, checks
the archive and module versions, and verifies that the container ID/start time
and running kernel have not changed. It never loads a module or restarts a
service. Needrestart uses listing mode. Each node runs the operation in a
transient systemd unit with 3 GiB memory, no swap, one CPU's quota, low priority
and idle-class I/O. Logs remain at `/var/log/spark3-prepare-kernel64k.log`.

Native inventories and installation logs are retained in `runs/`, including
unsuccessful attempts. Results describe preparation only. There is no measured
memory saving or performance result until a coordinated trial boot occurs.

## Doctor

`bin/spark3 doctor --live` reads the policy and checks:

- current kernel, actual base page size, loaded driver, THP policy and free-memory
  reserve alignment across all nodes;
- exact candidate package versions, both kernel images/configs/initramfs files,
  candidate headers and its 64 KiB build configuration;
- NVIDIA, ConnectX and fan-control module vermagic, NVIDIA module versions and
  fan-control DKMS installation, plus a signed fan module and enrolled signing
  certificate when Secure Boot is enabled;
- a candidate GRUB entry, an explicit normal-boot fallback and any pending
  one-shot or recovery override.

Missing candidate artifacts are warnings; inconsistent live host facts fail
doctor. Suggested corrections are printed separately, never executed. Different
node filesystem UUIDs in GRUB identifiers do not count as drift. A successful
check means prepared, not boot-tested. Filesystem existence checks do not prove
an initramfs is bootable; the preparation receipt additionally checks that the
archive can be listed and contains modules for the target release.

## Future trial (not performed by preparation)

Coordinate with the cluster holder and obtain permission to stop/reboot serving.
Capture a fresh 4 KiB baseline, then trial one node before moving the whole
cluster to 64 KiB. Keep the same image, serving configuration, KV budget and
pinned verification costs. Check large GPU mappings, pinned-buffer I/O, Engram,
RoCE, CUDA graphs and output correctness before benchmark comparisons.

Handle the 64 KiB swap format separately; never reformat the existing 4 KiB
swap file in place. Record THP and memory-watermark policy so reserve changes
are not confused with base-page metadata savings. Use an explicit GRUB
one-shot entry only in the authorized trial; ordinary reboot remains on the old
kernel. A one-shot entry does not restart a hung machine: verify recovery access.
Return to 4 KiB for the final control. Promote only after measured validation.

## Preparation result

Completed on dgx1 at 05:19:49 UTC and dgx2/dgx3 at 05:21:38 UTC on
2026-10-02. All four package versions match, as do the candidate kernel image
and configuration SHA-256 hashes. Both old and new boot artifacts exist;
candidate initramfs archives list successfully and contain target modules.
NVIDIA, ConnectX and fan-control modules match the target kernel, and the fan
signing certificate is enrolled on all three Secure Boot-enabled nodes.

All nodes still run 4 KiB pages on the original kernel. Their original serving
container IDs and start times are unchanged. No one-shot boot is armed. The
new collector reports no preparation findings or cross-node alignment issues;
the complete `doctor --live` passes. Host-independent validation: 49 existing
doctor tests and 14 kernel-readiness tests pass. No upstream patch or serving
image changed. Raw inventories, installation logs and receipts are in `runs/`.
