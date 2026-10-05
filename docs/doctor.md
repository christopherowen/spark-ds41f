# `bin/spark doctor`

`doctor` checks that a cluster configuration, the repository and, with
`--live`, the nodes themselves match what a deployment needs. It only reads:
it never installs, restarts or changes anything. Where a finding has a fix,
the fix is printed as a command for you to run.

Select a profile the same way as for every other command:

```sh
bin/spark doctor                                           # config/cluster.json
bin/spark --cluster-config config/cluster-tp4.json doctor  # the TP4 profile
```

## Modes

| Command | Runs where | Needs | Checks |
| --- | --- | --- | --- |
| `bin/spark doctor` | any checkout (CI runs it) | nothing outside the repository | [configuration](#configuration-checks) |
| `bin/spark doctor --live` | the head node | SSH to every node as `ssh_user`, passwordless `sudo -n` there | configuration first; if it is clean, every [live check](#live-checks) on every node, then [checkpoint fragmentation](#checkpoint-fragmentation) |
| `bin/spark doctor --fragmentation-commands` | the head node | SSH | prints `e4defrag` commands for fragmented checkpoint shards; never runs them |

Output is one line per finding, then a summary:

```text
WARN: dgx4: 46/48 weight shards are not at ext4's best extent count (...)
configuration OK; live cluster matches with 1 warning
```

`ERROR:` lines fail the command (exit status 1). `WARN:` lines never fail it
and never block a start or a benchmark. They mark something that costs memory
or latency, or that could fail later.

Before the first start, `--live` reports `cannot inspect dsv41-karmic-kraken`
on every node, and with the service stopped it reports `container is not
running`. Those two are expected until the service runs; read the rest of the
output for setup problems.

## Configuration checks

These read only the selected cluster configuration, its node map, the lock
(`upstreams.lock.json`, or the configuration's `upstreams_config`) and the
repository. Every one is an error unless marked **warn**.

| Area | Fails when |
| --- | --- |
| Node map | not 3 or 4 nodes; ranks or names repeat; not exactly one API head, at rank 0; `distributed.master_addr` is not the head's `management_ip` |
| Fabric | the transport is unknown or does not fit the node count (four nodes need a ring transport); `roce_peer_hcas` does not name the right peers in cable order; a cable's stripes are not reciprocal; two peers share a local HCA; a `roce_subnets` entry is not its cable's /24, or a subnet does not connect exactly the two cable ends; `roce_gid_index` is not a nonnegative integer |
| Serving arguments | `--tensor-parallel-size`, `--nnodes` or the drafter's TP differ from the node count; the transport's required settings are missing; a sparknet transport carries B12X `B12X_ROCE_*` settings or the reverse |
| Kernel backend | `--attention-backend`, `--linear-backend`, `--moe-backend`, the drafter's attention backend or `VLLM_DS41_KERNEL_BACKEND` disagree with `kernel_backend` (absent means B12X) |
| Baseline | the configuration and the lock name different baselines |
| Benchmark reference | a `benchmark_reference` is set but is not a repository-relative path to an existing report, or that report was measured on another node count, transport or `kernel_backend`; **warn** when the report does not record one of them |
| Memory guards | a guard value is missing or not positive; the startup guard is not larger than the steady guard |
| Deployment | `repository`, `path`, `ready_url` or `ready_timeout_seconds` is empty; `launch_enabled` is not true or false |
| Sources | a revision is not a full commit SHA; a tracking ref or upstream is missing; a patch series or patch file is missing; a patch set's fingerprint differs from the source manifest; TileLang is listed without TileKernels (or the reverse) |
| Image identity | the image's expected tree labels do not match the source manifest; a sparknet transport runs on an image without sparknet, or a sparknet image is given a B12X transport |
| Secrets | a tracked file is named `.env`, `id_rsa`, `id_ed25519` or `credentials.json`, or ends in `.pem`, `.key`, `.local.env` or `.local.json` |
| Watchlist | **warn**: `docs/inspiration.md` is missing, or a watched reference is absent from it |

Fix these in the repository; nothing on the nodes changes them.

## Live checks

`--live` runs these on every node in the node map. The fixes below are the
ones `doctor` prints; [setup.md](setup.md) explains each change and how to
revert it.

### Host

| Check | Severity | Why | Fix |
| --- | --- | --- | --- |
| Boot target is `multi-user.target` | warn | the desktop's processes and GPU buffers come out of the model's unified memory | `sudo systemctl set-default multi-user.target` |
| Display manager is not running | error | same | `sudo systemctl disable --now display-manager.service` |
| Installed OS is first in the UEFI boot order | warn | a network (PXE) entry first adds about a minute of DHCP timeouts to every boot | the printed `sudo efibootmgr --bootorder …` |
| `nvidia-drm` kernel mode setting (`modeset=Y`) | error | the display carve-out that holds the embedding and output head needs it (the worker refuses to start without it), and a display attached after boot stays dark | `sudo apt purge nvidia-drm-options-modeset0`, then reboot with the service stopped |
| `nvidia-drm` framebuffer console (`fbdev=Y`) | error | an attached display gets no text console | remove the `fbdev=0` option from `/etc/modprobe.d`, `sudo update-initramfs -u`, reboot |
| A DRM card belongs to the nvidia driver | error | the display carve-out allocates through it | check that `nvidia-drm` loaded with `modeset=Y` (`ls -l /dev/dri/by-path`) |
| No process holds DRM master | error (unreadable: warn) | the text console stops drawing while one does | stop or restart that process |
| Active console is in text mode | error | an attached display shows nothing | `scripts/host-recovery apply` |
| Kernel command line has no `splash` | warn | Plymouth leaves the console in graphics mode after boot | `scripts/host-recovery apply`, then reboot with the service stopped |
| Unused desktop services | warn | Bluetooth, CUPS, snapd, fwupd's refresh timer and the DGX Dashboard hold memory; the dashboard's update checks peak near 850 MiB | `sudo systemctl disable --now …` (`scripts/host-recovery apply` does the same) |
| NVMe interrupt coalescing off | warn | Engram's small direct disk reads wait for the coalescing timer | `sudo systemctl mask --now nvidia-nvme-interrupt-coalescing.service && sudo /usr/bin/nvidia-nvme-interrupt-coalescing.sh disable` |
| Fan-floor control: DKMS module, loaded module, cooling device, daemon | error, first missing layer | `bin/spark bench` and kernel-lab jobs pre-cool through it; serving does not need it | the printed `dkms autoinstall`, `modprobe` or `systemctl enable --now` command ([dgx-spark-fan-control](https://github.com/christopherowen/dgx-spark-fan-control)) |

### Kernel and memory saver

These compare each node with the kernel policy in
[config/kernel-trial.json](../config/kernel-trial.json): the 4 KiB kernel
`7.0.0-1019-nvidia`, the 64 KiB kernel `7.0.0-1019-nvidia-64k`, NVIDIA
580.178.04 and memory-saver 0.2.0. Corrections are collected at the end under
`Kernel corrections (not executed)`.

| Check | Severity |
| --- | --- |
| The running kernel and page size are one of the prepared pair | warn |
| Loaded NVIDIA driver, 64 KiB kernel package versions, kernel images and initramfs for both kernels, 64 KiB headers and `CONFIG_ARM64_64K_PAGES` | warn |
| `cpupower` for the 64 KiB kernel and the `performance` CPU governor | warn |
| NVIDIA, ConnectX and fan-control modules built for the 64 KiB kernel; fan-control DKMS installed for it; with Secure Boot, a signed fan module and an enrolled key | warn |
| A GRUB entry for the 64 KiB kernel, normal boot pinned to the policy's default, and no pending one-shot or recovery override | warn |
| On 64 KiB: THP `never`, `vm.min_free_kbytes=45166`, `/swap-64k.img` active and `spark3-kernel-memory.service` active | warn |
| Memory-saver DKMS installed for the 64 KiB kernel and selected for UVM; with Secure Boot, signed; on 64 KiB, loaded with leaf-table packing on | warn |
| **The selected profile's kernel requirements**: the running kernel and page size equal the profile's `host.kernel`; for a profile with `memory_saver_required`, the memory saver is loaded with packing on and the driver is 580.178.04 | error |
| All nodes agree on kernel, page size, driver, loaded NVIDIA module sources, packing, THP, free-memory reserve, Secure Boot state and CPU governors | error |

The 4 KiB profile (`config/cluster-4k.json`) requires only the 4 KiB kernel,
so on that profile the memory-saver lines stay warnings.

### Fabric

| Check | Severity | Fix |
| --- | --- | --- |
| Every RDMA device in `roce_peer_hcas` exists | error | check cabling and `rdma link` |
| The configured GID index (3 by default) holds the device's IPv4 RoCE v2 address | error | if the GID moved to another slot, re-activate the interface with `nmcli` while nothing uses RDMA |
| That address is in the cable's `roce_subnets` network | error | check the interface address and the cable it is on |

A GID moves when a link flaps while RDMA resources still hold the old entry;
NCCL then fails with an unhandled system error.

### Service

| Check | Severity |
| --- | --- |
| The serving container exists and is running | error |
| Its image tag, vLLM command line and environment equal the configuration's | error |
| The image's source-tree labels equal the configuration's `expected_labels` | error |
| While serving, the GPU clock is at least 1,000 MHz | error; a lower clock is the [GB10 clock latch](recovery.md#gb10-gpu-clock-latch), cleared only by removing AC power |

### Checkpoint fragmentation

**Warn** when model shards on a node use more extents than ext4's best layout
for their size. Fragmented shards take longer to read at startup. `bin/spark doctor
--fragmentation-commands` prints `sudo e4defrag -v <shard>` for each
fragmented file; run them with the service stopped.

## Launch preflight

`bin/spark cluster start --apply` repeats the checks a start cannot survive,
on every node, and refuses to launch on any failure:

- the configuration has `deployment.launch_enabled: true`;
- local `doctor` is clean and the local checkout has no uncommitted files;
- each node's checkout is at the same commit, clean, and its `origin` is the
  deployment repository;
- the profile's kernel requirements (above);
- the RDMA devices, GID index and cable subnets (above);
- `bin/spark doctor` passes on the node itself;
- every mount source exists (writable for read-write mounts);
- the image is present with the expected tree labels;
- no container with the serving name exists unless you pass `--replace`;
- passwordless `sudo` works (the memory guards need it).

Then it arms a 5 GiB startup memory guard on every node before any container
starts, starts the ranks, waits for the API, and switches to the 3 GiB steady
guard. A node whose available memory falls below the guard has its container
killed, which protects SSH and the host rather than an unready service.

`bin/spark bench` also refuses to measure a cluster whose live configuration
differs from the selected profile.
