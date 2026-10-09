# Setting up DeepSeek V4.1 Flash on DGX Spark

This guide takes three or four DGX Sparks from a fresh DGX OS install to the
promoted deployment: DeepSeek V4.1 Flash on the TileLang kernel family, with
tensor parallelism across the Sparks over direct-cabled ConnectX-7 links.

Every stage says what it changes on your machines, which profiles need it,
how to check it, and how to undo it. Read the [summary of host
changes](#what-changes-on-your-machines) before you start.

## Requirements and tested configurations

| Sparks | Cabling | Profile | Context | Sequences | KV per rank | Kernel | Memory saver | Status |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 3 | triangle | [config/cluster-4k.json](../config/cluster-4k.json) | 262,144 | 8 | 2.2 GiB | stock 4 KiB | not needed | supported fallback |
| 3 | triangle | [config/cluster.json](../config/cluster.json) (= `cluster-64k.json`) | 524,288 | 8 | 3.5 GiB | 64 KiB | required | promoted |
| 4 | ring | [config/cluster-tp4.json](../config/cluster-tp4.json) | 1,048,576 | 16 | 10.5 GiB | 64 KiB | required | promoted |
| 2 | — | — | — | — | — | — | — | not supported |

**Two Sparks** cannot hold this model with useful context. At three Sparks,
the head node already runs within about 3 GiB of its memory guard during a
500K prefill, and two would each hold half as much again of the weights.

**The 4 KiB profile is the shortest path to a running service**: it needs no
kernel change and no out-of-tree module. The 64 KiB profiles recover about
1.8 GiB per node with the [memory saver](https://github.com/christopherowen/dgx-spark-memory-saver),
which pays for their larger context.

Each node needs:

- a DGX Spark (GB10, 128 GB unified memory) on DGX OS 26.09.2 or later;
- Docker with the NVIDIA container runtime and `buildx`, and `/dev/infiniband`;
- about 480 GiB of free disk (the model, including Engram tables read from
  disk, plus caches); the build host also needs 64 GiB of free memory;
- a management network for SSH, the API and Gloo, with passwordless SSH from
  the head node to every node as one user, who also has passwordless `sudo`
  (the memory guards and several checks use `sudo -n`);
- the ConnectX-7 cabling for its topology ([stage 3](#3-fabric-cabling-and-addressing)).

## What changes on your machines

| Change | 4 KiB | 64 KiB, TP4 | Checked by | Revert |
| --- | --- | --- | --- | --- |
| [Host recovery policy](#host-recovery-policy) (watchdog, SSH/Tailscale protection, text console) | recommended | recommended | `scripts/host-recovery check`, `doctor --live` | [remove the drop-ins](#revert-the-host-baseline) |
| [Headless boot](#headless-boot-the-trade-off) (no desktop, idle services off) | required by `doctor --live` | same | `doctor --live` | `sudo systemctl set-default graphical.target`, re-enable the services |
| [Kernel mode setting](#kernel-mode-setting) (`nvidia-drm modeset=Y`) | required | required | `doctor --live` | reinstall `nvidia-drm-options-modeset0` |
| [`vm.watermark_boost_factor=0`](#memory-accounting) | recommended | recommended | — | delete the sysctl file |
| [NVMe interrupt coalescing off](#nvme-interrupt-coalescing) | optional | optional | `doctor --live` (warn) | unmask the service |
| [UEFI boot order](#boot-order) | optional | optional | `doctor --live` (warn) | `efibootmgr --bootorder` |
| [Fan-floor control](#fan-floor-control) | recommended | recommended | `doctor --live` (error until installed) | its [uninstall guide](https://github.com/christopherowen/dgx-spark-fan-control/blob/main/docs/maintenance.md#uninstall) |
| [64 KiB kernel, swap and memory policy](#2-kernel-64-kib-profiles) | — | required | `doctor --live`, launch | [boot the 4 KiB kernel](#revert-the-64-kib-kernel) |
| [Memory saver](#2-kernel-64-kib-profiles) (signed DKMS UVM module) | — | required | `doctor --live`, launch | its [removal guide](https://github.com/christopherowen/dgx-spark-memory-saver/blob/main/docs/dkms.md#remove-and-restore-stock-uvm) |
| [ConnectX-7 addressing](#3-fabric-cabling-and-addressing) (netplan) | required | required | `doctor --live`, launch | remove `40-cx7.yaml` |
| Docker image and container | required | required | `doctor --live` | `bin/spark cluster stop --apply`, `docker image rm` |

The repository changes nothing on the hosts by itself. Each change is a
command you run, and `doctor` only reports and prints fixes ([doctor.md](doctor.md)).
`bin/spark cluster start` refuses to launch a profile whose kernel
requirements are not met, so a skipped step fails before any container
starts.

## Order of operations

Do the stages in order on every node, and keep the nodes identical:
`doctor --live` fails when they differ in kernel, driver or memory policy.

| Stage | What | Check before moving on |
| --- | --- | --- |
| 1 | [Host baseline](#1-host-baseline) | `scripts/host-recovery check` on each node |
| 2 | [64 KiB kernel and memory saver](#2-kernel-64-kib-profiles) (skip for the 4 KiB profile) | `getconf PAGESIZE` is 65536; packing is `Y` |
| 3 | [Fabric](#3-fabric-cabling-and-addressing) | 9000-byte pings on every path; GID index 3 is IPv4 |
| 4 | [Site configuration](#4-site-configuration) | `bin/spark doctor` |
| 5 | [Model](#5-model) | the pinned revision on every node |
| 6 | [Image](#6-image) | the same image ID on every node |
| 7 | [First start](#7-first-start) | `bin/spark doctor --live` |
| 8 | [Verify](#8-verify) | `bin/spark bench` |

Until stage 7, `doctor --live` also reports the missing container; that is
expected ([doctor.md](doctor.md#modes)).

## 1. Host baseline

### Host recovery policy

From a clean checkout on each node:

```sh
scripts/host-recovery apply
scripts/host-recovery check
```

It installs a narrow management-plane policy ([host/recovery](../host/recovery/README.md)):

- systemd feeds the SBSA hardware watchdog, which reboots only after a real
  system stall;
- SSH and Tailscale get a little reclaim protection and strongly negative OOM
  scores, and a once-a-minute timer restarts either if it stops answering;
- Plymouth's boot gate is bounded, and `splash` is removed from the kernel
  command line, so an attached display shows a text console;
- the display manager (`gdm3`) and unused desktop services are disabled
  ([headless boot](#headless-boot-the-trade-off)).

It never stops the serving container and never reboots on a network outage.
The `splash` change takes effect at the next reboot, and `check` fails until
then; reboot with the service stopped.

### Headless boot: the trade-off

The recipe runs every node without a graphical session:

```sh
sudo systemctl set-default multi-user.target
sudo systemctl disable --now display-manager.service   # host-recovery apply does this
```

**What you gain:** the desktop's processes and GPU buffers come out of the
same unified memory as the model. At TP3 the head node runs within about
3 GiB of its steady memory guard during a 500K-token prefill, so a desktop
session can push a long prefill into the guard, which kills the container.
`scripts/host-recovery apply` also disables Bluetooth, CUPS, snapd, fwupd's
refresh timer and the DGX Dashboard, whose update checks peak near 850 MiB.

**What you lose:** the GNOME desktop on an attached monitor (a text console
remains: kernel mode setting keeps it working), the DGX Dashboard web UI,
Bluetooth, printing and snap applications.

**To get them back:**

```sh
sudo systemctl set-default graphical.target
sudo systemctl enable --now gdm3.service
sudo systemctl enable --now dgx-dashboard.service dgx-dashboard-admin.service
sudo systemctl enable --now bluetooth.service cups.service snapd.service   # as needed
```

`doctor --live` then reports the display manager as an error and the
services as warnings; serving still starts, with less memory margin.

### Kernel mode setting

DGX OS ships `nvidia-drm-options-modeset0`, which turns `nvidia-drm` mode
setting off. The promoted profiles load the embedding and output head into
the GB10 display carve-out (`SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`), which needs
mode setting, and a display attached after boot stays dark without it:

```sh
sudo apt-get purge nvidia-drm-options-modeset0
sudo update-initramfs -u -k all
sudo reboot   # with the service stopped
```

Revert: `sudo apt-get install nvidia-drm-options-modeset0` and reboot; the
carve-out profiles will then refuse to start.

### Memory accounting

The memory guards read MemAvailable. With the kernel default, a watermark
boost hides up to 0.87 GiB of it:

```sh
echo "vm.watermark_boost_factor = 0" | sudo tee /etc/sysctl.d/90-watermark-boost.conf
sudo sysctl -w vm.watermark_boost_factor=0
```

Revert: delete the file and `sudo sysctl -w vm.watermark_boost_factor=15000`.

### NVMe interrupt coalescing

DGX OS turns on NVMe interrupt coalescing at every boot (one interrupt per 8
completions or 100 µs). Engram's small disk reads then wait for the timer
([measurement](../experiments/2026-10-02-nvme-coalescing/README.md)):

```sh
sudo systemctl mask --now nvidia-nvme-interrupt-coalescing.service
sudo /usr/bin/nvidia-nvme-interrupt-coalescing.sh disable
```

Revert: `sudo systemctl unmask nvidia-nvme-interrupt-coalescing.service` and
reboot.

### Boot order

A firmware update can leave a network (PXE) entry ahead of the installed OS,
adding about a minute of DHCP timeouts to every boot. `doctor --live` prints
the `sudo efibootmgr --bootorder …` that puts Ubuntu first.

### Fan-floor control

[dgx-spark-fan-control](https://github.com/christopherowen/dgx-spark-fan-control)
adds a fan floor the EC enforces on top of NVIDIA's curve
([installation](https://github.com/christopherowen/dgx-spark-fan-control/blob/main/docs/installation.md)). Serving does not use
it, but `doctor --live` reports it as an error until it is installed.
`bin/spark bench` and kernel-lab jobs use it to cool every node below 55 °C
before measuring; without it they wait for the nodes to cool on NVIDIA's
curve. With Secure Boot, its installation creates and enrolls a MOK signing
key, and the memory saver's install script in [stage 2](#2-kernel-64-kib-profiles)
signs with the same key.

Install its `dgx-fan-control.service` as well; `doctor --live` reports an
error until it is installed. `bin/spark bench` requires it: it starts the
service's performance curve on every node while it measures, so every
benchmark runs under the same fan policy, and stops it again where it started
it. Whether it also runs while serving is your choice. The curve raises the fan floor from 50 °C, well before
NVIDIA's curve, so nodes run cooler and louder under load; serving does not
need it. Pick either
[startup mode](https://github.com/christopherowen/dgx-spark-fan-control/blob/main/docs/installation.md#5-choose-your-startup-mode),
the same on every node.

### Check

On each node, `scripts/host-recovery check` passes and
`systemctl --failed` lists nothing. `kho=off` is on the kernel command line
(`grep -o kho=off /proc/cmdline`); DGX OS 26.09.2 adds it through
`nvidia-spark-grub-kho`. Without it, RDMA memory registration fails under
memory pressure on the 7.0 kernels ([recovery.md](recovery.md#kernel)).

### Revert the host baseline

`scripts/host-recovery` has no remove action. To remove the policy, delete
what it installed, then `sudo update-grub`, `sudo systemctl daemon-reload` and
reboot:

- `/etc/systemd/system.conf.d/95-spark-recovery.conf`
- `/etc/sysctl.d/95-spark-recovery.conf`
- `/etc/ssh/sshd_config.d/95-spark-recovery.conf`
- `/etc/systemd/system/{ssh,tailscaled,plymouth-quit-wait}.service.d/95-spark-recovery.conf`
- `/usr/local/libexec/spark-management-health` and
  `/etc/systemd/system/spark-management-health.{service,timer}` (disable the
  timer first)
- `/etc/default/grub.d/zz-spark-console.cfg`

## 2. Kernel (64 KiB profiles)

Skip this stage for `config/cluster-4k.json`.

The 64 KiB profiles run Ubuntu's `7.0.0-1019-nvidia-64k` kernel with NVIDIA
580.178.04 and the memory saver, a patched, signed `nvidia-uvm` DKMS module
that stops GPU page tables wasting whole 64 KiB pages. The exact package
versions, driver and memory-saver revision are pinned in
[config/kernel-trial.json](../config/kernel-trial.json), and `doctor --live`
checks each against it. The repository checks these; installing them is
yours to run, as root, with the service stopped:

1. **Kernel packages.** Install the five `7.0.0-1019-nvidia-64k` packages at
   the versions in `config/kernel-trial.json` (image, modules, headers,
   `linux-tools` for the CPU governor, and `linux-modules-nvidia-580-open`).
   Keep the 4 KiB kernel installed: it is the fallback. With fan control
   installed, also run `sudo dkms autoinstall -k 7.0.0-1019-nvidia-64k` so
   its module is built for the new kernel. The memory saver's [kernel guide](https://github.com/christopherowen/dgx-spark-memory-saver/blob/main/docs/kernel.md)
   lists the same steps.
2. **Swap and memory policy.** Run
   `sudo experiments/2026-10-02-kernel-64k/install-boot-support.sh`. It adds a
   16 GiB `/swap-64k.img` with a 64 KiB header, marks the original
   `/swap.img` `noauto`, and enables `spark3-kernel-memory.service`, which at
   boot activates the swap file that matches the running page size and, on
   64 KiB only, sets THP to `never` and `vm.min_free_kbytes=45166`
   ([host/kernel-memory](../host/kernel-memory/README.md)). It does not
   reboot or touch the running system.
3. **Memory saver.** Run `bash experiments/2026-10-02-memory-saver-capacity/install-dkms.sh`
   from a clean checkout. It clones the memory saver at the revision pinned
   in `config/kernel-trial.json`, then builds and installs DKMS 0.2.0 for the
   64 KiB kernel. It signs with the key that dgx-spark-fan-control created
   (`/root/.local/share/dgx-spark-fan-control/keys`) and refuses to run unless
   that key is enrolled in MOK. Without fan control, follow the memory
   saver's [DKMS guide](https://github.com/christopherowen/dgx-spark-memory-saver/blob/main/docs/dkms.md), which sets up your own
   enrolled key.
4. **Trial boot.** On one node, boot the 64 KiB kernel once, leaving GRUB's
   default alone:

   ```sh
   uuid=$(findmnt -n -o UUID /)
   sudo grub-reboot "gnulinux-advanced-$uuid>gnulinux-7.0.0-1019-nvidia-64k-advanced-$uuid"
   sudo reboot
   ```

   Check it (below); any later reboot returns to the default. Then repeat on
   the other nodes.
5. **Make it the default.** Pin GRUB's normal default, so that a later kernel
   installation cannot silently change the boot kernel:

   ```sh
   uuid=$(findmnt -n -o UUID /)
   echo "GRUB_DEFAULT=\"gnulinux-advanced-$uuid>gnulinux-7.0.0-1019-nvidia-64k-advanced-$uuid\"" |
     sudo tee /etc/default/grub.d/zz-spark-kernel-trial.cfg
   sudo update-grub
   ```

### Check

On every node:

```sh
getconf PAGESIZE                                                 # 65536
cat /sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables   # Y
swapon --show                                                    # /swap-64k.img
```

Then from the head node, `bin/spark doctor --live` must show no kernel or
memory-saver findings, and no `nodes differ` line.

### Revert the 64 KiB kernel

Stop the service. On every node, put the retained 4 KiB entry in the pin
file (`gnulinux-advanced-$uuid>gnulinux-7.0.0-1019-nvidia-advanced-$uuid`),
run `sudo update-grub` and reboot. The memory service activates the original swap and leaves THP alone,
and the memory saver is not built for the 4 KiB kernel, so stock UVM loads.
Serve with `--cluster-config config/cluster-4k.json`. To remove everything,
follow the memory saver's [removal guide](https://github.com/christopherowen/dgx-spark-memory-saver/blob/main/docs/maintenance.md),
disable `spark3-kernel-memory.service`, restore
`/var/lib/spark3/kernel-64k-boot/fstab.before`, and delete `/swap-64k.img`
once it is inactive ([host/kernel-memory](../host/kernel-memory/README.md)).
Never remove an active swap file.

## 3. Fabric: cabling and addressing

Each Spark has one ConnectX-7 card with two 200 GbE ports. Each port carries
two PCIe paths (for example `enp1s0f0np0` and `enP2p1s0f0np0`), so one cable
between two Sparks gives two RDMA paths.

| Topology | Cables | Cabling | Transport |
| --- | ---: | --- | --- |
| 3 Sparks, triangle | 3 | every pair cabled directly: port 0 to the next node, port 1 to the previous | `oneshot-direct` |
| 4 Sparks, ring | 4 | each node to the next and the previous, `1-2-3-4-1` | `oneshot-ring4` |

No switch is involved. In the ring each rank exchanges data only with its
previous and next ranks, so the ranks in the node map must follow the cable
order.

**Addressing.** Give every cable path its own /24 subnet,
`10.<a><b>.<path>.<node>`: the dgx1-dgx2 cable's first path is `10.12.1.1` on
dgx1 and `10.12.1.2` on dgx2, its second path `10.12.2.x`. Use MTU 9000. The
addresses live in a netplan file such as `/etc/netplan/40-cx7.yaml`.
[sparknet](https://github.com/christopherowen/dgx-spark-networking) generates
both the node map and each node's netplan from the actual cabling. Install
its CPU tooling on the head node (`pip install` from a clone; see its
[integration guide](https://github.com/christopherowen/dgx-spark-networking/blob/main/docs/integration.md#21-install)).
Discovery reads each node's LLDP neighbours over SSH with `sudo -n lldpctl`,
so every node needs `lldpd` running. It changes nothing on a host:

```sh
sparknet topology discover dgx1 dgx2 dgx3 --out site --management-ip dgx1=<ip> ...
```

Review the generated files, install each node's `40-cx7.yaml` and run
`sudo netplan apply`. [sparknet's topology guide](https://github.com/christopherowen/dgx-spark-networking/blob/main/docs/topology.md)
and [host prerequisites](https://github.com/christopherowen/dgx-spark-networking/blob/main/docs/host-prerequisites.md)
cover the details.

### Check

From each node, a 9000-byte ping reaches both ends of every cable path:

```sh
ping -c2 -M do -s 8972 10.12.1.2
```

and each RoCE device's GID index 3 holds its IPv4 address
(`cat /sys/class/infiniband/<device>/ports/1/gids/3` ends in the address in
hex). `doctor --live` and the launch preflight check both.

Revert: remove (or restore) the netplan file and `sudo netplan apply`.
Switching between a triangle and a ring is a recabling plus new addressing for
the cables that moved; re-run the discovery.

## 4. Site configuration

Two files describe your site and stay out of git:

- **Node map.** `config/nodes.json` for the three-node profiles (start from
  [config/nodes.example.json](../config/nodes.example.json)), or
  `config/nodes-ring4.local.json` for the TP4 profile (start from
  [config/examples/nodes-ring4.json](../config/examples/nodes-ring4.json)).
  It names each node, its rank (the head is rank 0), management IP,
  `ssh_user`, and `roce_peer_hcas`: for each peer rank, the local RoCE
  devices cabled to it (`rdma link` shows them). `sparknet topology discover`
  writes it. Two optional fields per node change how the head reaches it:
  `ssh_host` replaces the node name for every SSH call, and `transfer_host`
  only for the lab's bulk copies (kernel bundles, their inputs, profiler
  traces). Setting `transfer_host` to the node's name on a CX7 link cabled
  to the head (the `dgxN-internal` names in `/etc/hosts`) keeps large copies
  off a struggling LAN switch; the head's `known_hosts` must know that name.
  A node the head is not cabled to keeps its LAN route.
- **The profile.** In the profile you will run (`config/cluster.json`, its
  named copies, or `config/cluster-tp4.json`), set
  `distributed.master_addr` to the head's management IP, `host.home`,
  `deployment.repository` if you use a fork (the launch checks every node's
  `origin` against it), and the interface names in `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`,
  `TP_SOCKET_IFNAME` and `NCCL_IB_HCA`. Keep `distributed.master_port` below
  the head's ephemeral port range (`sysctl net.ipv4.ip_local_port_range`);
  inside it, an outgoing connection can take the port and the head fails to
  start.

`bin/spark cluster sync --apply` moves every node's checkout to the head's
commit and copies the head's node map to the others.

For any profile other than `config/cluster.json`, put `--cluster-config` before
the subcommand of every command, for example
`bin/spark --cluster-config config/cluster-tp4.json doctor`.

### Check

`bin/spark doctor` reports `configuration OK`.

## 5. Model

On every node, download the pinned revision into the Hugging Face cache the
profile mounts:

```sh
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

## 6. Image

On one idle node (the build host), from a clean checkout of `main`:

```sh
bin/spark build prepare
bin/spark build image --apply
```

`build prepare` fetches the pinned vLLM, B12X, NCCL, TileLang, TileKernels
and sparknet sources, applies the patch series and checks every result
against the recorded trees. `build image --apply` refuses to run next to a
live service, sizes its compile jobs to available memory, never overwrites
an existing tag, and runs a GPU import smoke test. The first build takes
longer because it builds TileLang and TileKernels. More detail is in
[docker/README.md](../docker/README.md).

Copy the image to the other nodes and confirm one image ID everywhere:

```sh
for host in dgx2 dgx3; do            # add dgx4 for TP4
  docker save vllm-ds41f-kkref:04c30fa98e79-r6 | ssh "$host" docker load
done
for host in dgx1 dgx2 dgx3; do       # add dgx4 for TP4
  ssh "$host" docker image inspect vllm-ds41f-kkref:04c30fa98e79-r6 --format '{{.Id}}'
done
```

Your image ID differs from ours because the image records the deployment
commit. The source-tree labels must match the profile's `expected_labels`;
`doctor --live` and the launch preflight check them.

Revert: `docker image rm vllm-ds41f-kkref:04c30fa98e79-r6` on each node.

## 7. First start

The first start JIT-compiles kernels into `cache/kkref/jit/`, so it takes
longer than later ones. Prebuild FlashInfer's sampling module on each node so
that compile happens outside the memory-guarded startup:

```sh
experiments/2026-09-23-canonical-minimal/prebuild_flashinfer.sh \
  vllm-ds41f-kkref:04c30fa98e79-r6 kkref/flashinfer
```

Then, from the head node, with a clean checkout of a published commit:

```sh
bin/spark cluster sync --apply
bin/spark cluster start --apply
bin/spark doctor --live
```

For TP4, give each of these `--cluster-config config/cluster-tp4.json` before
the subcommand; `sync` then reaches the fourth node too.

`cluster start` runs the [launch preflight](doctor.md#launch-preflight),
arms a 5 GiB startup memory guard on every node, starts the ranks, waits for
the API, then switches to the 3 GiB steady guard. If a guard trips, the
launch rolls back and each rank's log is kept under
`results/private/failed-starts/`.

The API is OpenAI-compatible on the head node's port 8000, model
`deepseek-v4.1-flash`. Reasoning is on by default; pass
`"chat_template_kwargs": {"thinking": false}` to turn it off per request.

Stop: `bin/spark cluster stop --apply`. It stops the exact containers and
their memory guards and keeps the containers for inspection.

## 8. Verify

From the head node, against the running, otherwise idle service:

```sh
bin/spark bench
```

It checks that the live cluster matches the profile, runs the quality gate
and the decode matrix (about six minutes), and compares the result with the
profile's promoted reference run, which has the same node count, transport and
kernel backend. `--full` runs every suite to tighter intervals (about
35 minutes). It exits non-zero if the quality gate fails, any request fails,
or a point is significantly slower than the reference (Welch 95% interval) by
more than 3%. See the README's [Benchmarking](../README.md#benchmarking)
section.

## When something fails

- [doctor.md](doctor.md) lists every check, what a failure means and its fix.
- [recovery.md](recovery.md) covers host incidents: a node that stops
  answering SSH, the boot gate, and the GB10 clock latch.
- [memory-profiles.md](memory-profiles.md) covers the page-size profiles in
  more depth.
- [switchless-topology.md](switchless-topology.md) covers three- and
  four-node fabrics and their transports.
