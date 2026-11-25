# `bin/spark doctor`

`doctor` checks that a cluster configuration, the repository and, with
`--live`, the nodes themselves match what a deployment needs. It only reads:
it never installs, restarts or changes anything. Where a finding has a fix,
the fix is printed as a command for you to run.

Select a profile the same way as for every other command:

```sh
bin/spark doctor                       # config/cluster.json at the site's shape
bin/spark --shape tp3-triangle doctor  # the recipe at another shape
```

## Modes

| Command | Runs where | Needs | Checks |
| --- | --- | --- | --- |
| `bin/spark doctor` | any checkout (CI runs it) | nothing outside the repository | [configuration](#configuration-checks) |
| `bin/spark doctor --live` | the head node | SSH to every node as `ssh_user`, passwordless `sudo -n` there | configuration first; if it is clean, every [live check](#live-checks) on every node, then [checkpoint fragmentation](#checkpoint-fragmentation) |
| `bin/spark doctor --live --stream-rate` | the head node | as `--live`, the profile's image on every node, and GPUs you may use | every live check, then the [memory read rate](#memory) of every node whose GPU is idle |
| `bin/spark doctor --fragmentation-commands` | the head node | SSH | prints `e4defrag` commands for fragmented checkpoint shards; never runs them |

Output is one line per finding, then a summary:

```text
WARN: dgx4: 46/48 weight shards are not at ext4's best extent count (...)
configuration OK; live cluster matches with 1 warning
```

`ERROR:` lines fail the command (exit status 1). `WARN:` lines never fail it
and never block a start or a benchmark. They mark something that costs memory
or latency, or that could fail later.

Before the first start, `--live` reports `cannot inspect spark-ds41f`
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
| Fabric | `fabric.transport` is missing (configurations recorded before r6 ran B12X's retired `rocenante-direct`) or unknown; the node map or the environment fails sparknet's checks at the lock's revision (`sparknet.topology`: the transport does not fit the node count, routes do not name the right peers in cable order or are not reciprocal, two peers share a local HCA, a subnet is not its cable's /24 or does not join exactly the two cable ends, the ring's NCCL policy is changed, and from the release after 0.5.0, a fabric setting the renderer owns is written into the environment) |
| Serving arguments | `--tensor-parallel-size`, `--nnodes` or the drafter's TP differ from the node count; the transport's required settings are missing; `nccl-ring` carries sparknet's `SPARKNET_ROCE_*` settings; the CUDA-graph capture sizes do not end at `--max-cudagraph-capture-size`, which vLLM refuses at startup (the finding names `sizing.max_num_seqs`, from which the recipe derives both); **warn** when the largest capture size is below `--max-num-seqs` x (1 + drafts), so full decode steps run without a CUDA graph |
| Kernel backend | `--attention-backend`, `--linear-backend`, `--moe-backend`, the drafter's attention backend or `VLLM_DS41_KERNEL_BACKEND` disagree with `kernel_backend` (absent means B12X); a TileLang configuration omits `VLLM_DS41_KERNEL_BACKEND` while its vLLM series has the switch, or sets it once the series records `tilelang-only` (the switch is gone) |
| Baseline | the configuration and the lock name different baselines |
| Qualified recipe | **warn** when a multi-node profile drifts from its stack's qualified settings: when the vLLM series includes L2 prefetch (`0781-`) without `VLLM_DS41_L2_PREFETCH_{WO,FFN,NEXT}_MB` set to `15`/`12`/`12` (or an overlaid `l2_prefetch.py` with `15`/`12`/`12` defaults), when `0806-` is present without the shape's `VLLM_DS41_L2_PREFETCH_WO_BANDS` (`17-32=12` for TP3, `17-32=10` for TP4–TP8), when Sparknet `0002-` is present without `SPARKNET_ROCE_MAX_COLLECTIVE_BYTES` (`2097152` for `tp4-ring`, `4194304` for TP3 shapes), or when `environment` (excluding rendered fabric variables and `VLLM_SERVER_DEV_MODE=1`) or `serve_args` differ from the shape's canonical profile |
| Benchmark reference | a `benchmark_reference` is set but is not a repository-relative path to an existing report, or that report was measured on another node count, transport or `kernel_backend`; **warn** when the report does not record one of them |
| Memory guards | a guard value is missing or not positive; the startup guard is not larger than the steady guard; `host.memguard_warn_gib` is not a non-negative integer, or is not 0 and not larger than the steady guard; `host.defragment_before_start` is set but is not true or false |
| Deployment | `repository`, `path`, `ready_url` or `ready_timeout_seconds` is empty; `launch_enabled` is not true or false |
| Sources | a revision is not a full commit SHA; a tracking ref or upstream is missing; a patch series or patch file is missing; a patch set's fingerprint differs from the source manifest; TileLang is listed without TileKernels (or the reverse) |
| Image identity | the image's expected tree labels do not match the source manifest; a sparknet transport runs on an image without sparknet; a sparknet configuration that pins no GID slot runs on a sparknet before 0.5.0 |
| GLP steering | a `glp` field is malformed: `repo` is not `<owner>/<name>`, `revision` not a full commit SHA, `file` not a `.gguf` name, `sha256` not 64 hex digits; `glp.served_model_name` is not served exactly once after the first `--served-model-name` (the first is the stock model every tool and probe uses); `WEIGHTLESS_STEER_*` is set without a `glp` block |
| Secrets | a tracked file is named `.env`, `id_rsa`, `id_ed25519` or `credentials.json`, or ends in `.pem`, `.key`, `.local.env` or `.local.json` |
| Watchlist | **warn**: `docs/inspiration.md` is missing, or a watched reference is absent from it |

A candidate whose vLLM manifest records `ds41-serving-contract` rejects
`VLLM_DS41_ENGRAM_OVERLAP`, `VLLM_DS41_MARKOV_NVFP4` and
`VLLM_DS41_DRAFT_NVFP4_HEAD`, including explicit `0` values. The first setting
has no consumer; the NVFP4 head choices are kept as an unselected experiment.
The candidate's vLLM startup rejects them too. Older frozen source recipes keep
their own option support.

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
| `nvidia-drm` kernel mode setting (`modeset=Y`) | error for legacy `/dev/dri/card0` carve-out profiles; warn on `1120+` (RM `/dev/nvidiactl`) or when off/unreadable | a display attached after boot stays dark without it; legacy pre-1120 DRM carve-out profiles also refuse to start without it | `sudo apt purge nvidia-drm-options-modeset0`, then reboot with the service stopped |
| `nvidia-drm` framebuffer console (`fbdev=Y`) | error (unreadable: warn) | an attached display gets no text console | remove the `fbdev=0` option from `/etc/modprobe.d`, `sudo update-initramfs -u`, reboot |
| A DRM card belongs to the nvidia driver | error for legacy `/dev/dri/card0` carve-out profiles; warn on `1120+` (RM `/dev/nvidiactl`) | an attached display stays dark without it; legacy pre-1120 DRM carve-out profiles allocate through it | check that `nvidia-drm` loaded with `modeset=Y` (`ls -l /dev/dri/by-path`) |
| No process holds DRM master | error (unreadable: warn) | the text console stops drawing while one does | stop or restart that process |
| Active console is in text mode | warn | an attached display shows nothing | `scripts/host-recovery apply` |
| Kernel command line has no `splash` | warn | Plymouth leaves the console in graphics mode after boot | `scripts/host-recovery apply`, then reboot with the service stopped |
| Unused desktop services | warn | Bluetooth, CUPS, snapd, fwupd's refresh timer and the DGX Dashboard hold memory; the dashboard's update checks peak near 850 MiB | `sudo systemctl disable --now …` (`scripts/host-recovery apply` does the same) |
| NVMe interrupt coalescing off | warn | Engram's small direct disk reads wait for the coalescing timer | `sudo systemctl mask --now nvidia-nvme-interrupt-coalescing.service && sudo /usr/bin/nvidia-nvme-interrupt-coalescing.sh disable` |
| NVMe APST: no sleep state of 5 ms or more to wake before 1 s idle, nor of 20 ms or more before 30 s | warn | the first Engram read after a pause waits for the drive to wake (11 ms from PS3, 46 ms from PS4) | `nvme_core apst_primary_timeout_ms=2000 apst_secondary_timeout_ms=60000` ([setup](setup.md#nvme-power-states-apst)) |
| Onboard Wi-Fi radio off and control plane (`management_ip` / `control_ip`, `*_SOCKET_IFNAME`) not on a wireless interface | warn | the control plane sends packets every decode step; Wi-Fi tail latency stalls the step ([#712](https://github.com/christopherowen/spark-ds41f/issues/712)) | `sudo nmcli radio wifi off`, or move `control_ip` / `*_SOCKET_IFNAME` onto wired or ConnectX-7 links |
| Fan-floor control: DKMS module (version 0.1.4), loaded module, cooling device | warn, first missing layer | `bin/spark bench` and kernel-lab jobs pre-cool through it; serving does not need it | the printed `dkms autoinstall` or `modprobe` command ([dgx-spark-fan-control](https://github.com/christopherowen/dgx-spark-fan-control)) |
| `dgx-fan-control.service` installed | warn | `bin/spark bench` runs its performance curve while measuring. Without it the bench warns and measures on NVIDIA's curve, so its numbers are not comparable with a reference that ran the performance curve. Whether it also runs while serving is the site's choice and is not checked | the printed `install` and `daemon-reload` commands |
| `lldpd` running, on nodes cabled to each other | warn | `sparknet topology discover` asks each node's lldpd which port every cable ends on. Without it discovery reports every port as unresolved and cannot write the node map or the netplan addresses, which you then write by hand ([setup](setup.md#3-fabric-cabling-and-addressing)), and a later recabling is not detected. Serving, the launch preflight and bench do not use it | `sudo apt install lldpd` |

### Kernel and memory saver

These compare each node with the kernel policy in
[config/kernel-trial.json](../config/kernel-trial.json): the 4 KiB kernel
`7.0.0-1019-nvidia`, the 64 KiB kernel `7.0.0-1019-nvidia-64k`, NVIDIA
580.178.04 and memory-saver 0.4.1. Corrections are collected at the end under
`Kernel corrections (not executed)`.

| Check | Severity |
| --- | --- |
| The running kernel and page size are one of the prepared pair | warn |
| Loaded NVIDIA driver, 64 KiB kernel package versions, kernel images and initramfs for both kernels, 64 KiB headers and `CONFIG_ARM64_64K_PAGES` | warn |
| `cpupower` for the 64 KiB kernel and the `performance` CPU governor | warn |
| NVIDIA, ConnectX and fan-control modules built for the 64 KiB kernel; fan-control DKMS installed for it; with Secure Boot, a signed fan module and an enrolled key | warn |
| A GRUB entry for the 64 KiB kernel, normal boot pinned to the policy's default, and no pending one-shot or recovery override | warn |
| On 64 KiB: THP `never`, `vm.min_free_kbytes=45166`, `/swap-64k.img` active and `spark-ds41f-kernel-memory.service` active (`spark3-kernel-memory.service` counts as the same policy, with a rename warning) | warn |
| Memory-saver DKMS installed for the 64 KiB kernel and selected for UVM; with Secure Boot, signed; on 64 KiB, loaded with leaf-table packing on | warn |
| **The selected profile's kernel recommendations**: the running kernel and page size equal the profile's `host.kernel`; for a profile with `memory_saver_required`, the memory saver is loaded with packing on and the driver is 580.178.04 | warn |
| All nodes agree on kernel, page size, driver, loaded NVIDIA module sources, packing, THP, free-memory reserve, Secure Boot state and CPU governors | error |

The 64 KiB kernel and memory saver are recommended for the 1M-context profiles,
so `doctor --live` and `cluster start` warn when a node runs a 4 KiB kernel or
lacks the memory saver, while still requiring all nodes in the cluster to agree
with one another.

### Fabric

| Check | Severity | Fix |
| --- | --- | --- |
| Every RDMA device in `roce_peer_hcas` exists | error | check cabling and `rdma link` |
| Each device's port carries exactly one IPv4 RoCE v2 GID, in its cable's `roce_subnets` network, at any index (and none outside it, which NCCL could take instead) | error | check the interface address and the cable it is on |
| Any GID slot a library is still fixed to (`NCCL_IB_GID_INDEX`, an earlier sparknet's `SPARKNET_ROCE_GID_INDEX`) holds that IPv4 RoCE v2 GID | error | if the GID moved to another slot, re-activate the interface with `nmcli` while nothing uses RDMA |

The kernel files a re-added address in the next free slot, so a link flap while
RDMA resources still hold the old entry moves the IPv4 RoCE v2 GID (from 3 to 4
on dgx2 and dgx4 when dgx1 rebooted under a starting service). NCCL with
`NCCL_IB_GID_INDEX` unset and sparknet 0.5.0 find the GID by address; a library
fixed to the old slot fails (NCCL with an unhandled system error).

### Service

| Check | Severity |
| --- | --- |
| The serving container exists and is running | error |
| Its image tag and vLLM command line equal the configuration's | error |
| Its environment (`Config.Env`) equals the configuration's: each variable `cluster start` passes (the profile's `environment` plus the per-node values it adds) has its value, and every other variable is the container image's own with the image's value; an extra, missing or different variable fails | error |
| Its bind mounts (`HostConfig.Binds`) equal the configuration's `container.mounts`; an extra, missing or different mount fails | error |
| The running container's immutable image has the configuration's `expected_labels` | error |
| Every node's running image has the same ordered layer digests and runtime `Config` | error |
| While serving, the GPU clock is at least 1,000 MHz | error; a lower clock is the [GB10 clock latch](recovery.md#gb10-gpu-clock-latch), cleared only by removing AC power |

Image checks inspect the container's immutable image ID, so moving a tag cannot hide
an old running build. Docker's containerd and overlay2 stores may assign different
IDs to the same copied image; the layers and runtime configuration must still match.
Benchmarks validate their own captured image inspections and record each node's
image ID, layers, labels and content fingerprint alongside any validation errors.

### GLP steering file

A profile with a `glp` block serves the steered model from a file every node keeps
in its hub cache at the block's pinned revision. The check runs whether or not the
service is up.

A node without the file is a warning, not an error. The start then leaves the block
out on every rank (ranks must not differ), so the cluster serves the stock model
name only and a request for the steered name is refused as an unknown model. The
warning names the nodes and prints the `hf download` command for the pinned revision
([setup](setup.md#glp-steering-file)); `doctor --live` compares a running cluster
with the same profile, so one started without the file passes with the same warning.
A file that is present and whose sha256 is not `glp.sha256` is an error: the
engine would steer with something other than the pinned vector.

On the head, the profile's image then reads the file with its own GLP reader
(`python3 -m vllm.models.deepseek_v4_1.glp`), mounted at `/glp/vector.gguf` as the
engine mounts it, in a `docker run --rm --pull=never --network=none` with no GPU.
The engine reads the file while it builds the model, so a vector its reader refuses
(another `glp.spec_version` or `glp.hook_point`, for example) would stop the whole
engine, the stock model name included. A refusal is an error naming the image, the
file's `glp.spec_version` and `glp.hook_point`, and the reader's reason. The check is
skipped when the head lacks the image; the image checks report that.

### CPU pinning

When a profile pins serving to CPUs (`--cpuset-cpus=` in `container.docker_run_args`),
every node's running container must use exactly that set (Docker's
`HostConfig.CpusetCpus`). The set must also be that node's fast cores: those whose
`cpu_capacity` is above the midpoint of the smallest and largest, GB10's ten Cortex-X925
(5-9 and 15-19). A pin to slow cores, or a container started without the pin, is an
error. Profiles without a pin are not checked.

### Memory

How fast a GB10 reads memory depends on the state of the node's free memory, not only on the
hardware. dgx4 read 253 GB/s seventeen minutes after a boot and 230 to 236 GB/s after ten days
of engine boots, with the same kernel, driver, image and benchmark; a reboot restored it, and
two hours and an image build later it read 238 to 245
([experiments/2026-11-17-c1-floor-now](../experiments/2026-11-17-c1-floor-now/README.md)).
After the boot almost all of the free memory sat in the largest blocks (512 MiB on the 64 KiB
kernel); ten days later none of it did.

That is the rate of memory taken as it comes, which is what a kernel-lab bundle gets. The
engine is mostly spared: its loader soaks the small free fragments before it places the
weights, so they sit in blocks of 2 MiB or more on an old node and a fresh one alike. In a
traced boot the rank on a rebooted node was 2% faster in two kernels, at most 0.6% of a
single-stream step. So these checks say whether a timing taken on a node can be compared with
one taken on a fresh node. They are not a reason to reboot a serving node.

| Check | Severity | Why |
| --- | --- | --- |
| On a node whose GPU no process holds, at least half of the free memory is in the largest block size (`/proc/buddyinfo`). Not judged below 32 GiB free | warn | kernel-lab timings on the nodes without large free blocks read memory 7 to 9% slower. `cluster start` drops caches and compacts before a launch, which does not rebuild blocks that kernel allocations have split; a reboot does |
| With `--stream-rate`: the node reads cold 220 MiB buffers at 245 GB/s or more | warn | a freshly booted GB10 reads 249 to 254 GB/s, the slow state 229 to 237 |

The first check costs nothing: it reads one file. A node whose GPU a process holds is not
judged, since its free memory is what the engine left. The half is a line between the two
states that were measured (96% and 0%), not a measured threshold, and the share does not
predict the rate: dgx4 two hours after its reboot still had most of its free memory in 512 MiB
blocks and read 238 to 245 GB/s. `--stream-rate` is the measurement; the block share is the hint to
take it.

`--stream-rate` prints one line a node (`dgx4: memory reads at 253.6 GB/s`). It starts one
container of the profile's image on each node for about a second of GPU time (ten seconds
with the container's start), so it is only run when asked. It skips a node whose GPU a
process holds or whose kernel-lab GPU lock is taken (`memory read not measured (...)`): the
read would disturb the engine and be slowed by it. Inside someone else's window it is a GPU
job like any other: the hold file's rule applies.

The probe measures memory taken now, without the loader's placement. A running engine's
weights were placed when it started.

### Checkpoint fragmentation

**Warn** when model shards on a node use more extents than ext4's best layout
for their size. Fragmented shards take longer to read at startup. `bin/spark doctor
--fragmentation-commands` prints `sudo e4defrag -v <shard>` for each
fragmented file; run them with the service stopped.

## Launch preflight

`bin/spark --cluster-config <profile> cluster start --apply` repeats the checks
a start cannot survive, on every node, and refuses to launch on any failure:

- the profile is named on the command line (every `--apply` refuses the
  read-only default);
- the engine on the head node, running or stopped, has the profile's shape:
  node count, tensor parallelism, context (`--max-model-len`) and KV size
  (`--kv-cache-memory-bytes`). A start that changes any of them needs
  `--change-shape`. A running engine on a different number of nodes is refused
  even then: stop it with its own profile first;
- the configuration has `deployment.launch_enabled: true`;
- local `doctor` is clean and the local checkout has no uncommitted files;
- each node's checkout is at the same commit, clean, and its `origin` is the
  deployment repository;
- the profile's kernel requirements (above);
- the RDMA devices, their GIDs, any pinned GID slot and the cable subnets (above). On more than one node,
  every RoCE interface the profile uses needs its `roce_subnets` entry: without
  it the start could not tell a triangle from a ring;
- `bin/spark doctor` passes on the node itself;
- every mount source exists (writable for read-write mounts);
- the `glp` file, if the profile has one: where a node lacks it the start warns and
  serves the stock model only; where it is present it has `glp.sha256` and the
  image's own GLP reader accepts it on the head (above);
- the image is present with the expected tree labels;
- no container with the serving name exists unless you pass `--replace`;
- passwordless `sudo` works (the memory guards need it).

The plan, without `--apply`, opens with the live engine (state, shape, image and
the profile it was started from), the planned engine and whether the shape
changes, and says when `--apply` would refuse; `--verbose` adds each rank's
`docker run`. Each container is labelled `spark.cluster-config=<profile>`, so the
next plan can name the profile that stops it.

The plan also runs the checks above, which only read, so every problem `--apply`
would stop on shows before it. When a node's checkout, profile or image differs
from the head's, the list ends with the command that repairs them all:

```
fix: bin/spark --cluster-config <profile> cluster sync --apply gives every node this commit and its image
```

A profile that no commit carries (an ignored `*.local.json`) reaches no other
node: commit and publish it.

Then it arms a 5 GiB startup memory guard on every node before any container
starts, starts the ranks, waits for the API, and switches to the 3 GiB steady
guard. A node whose available memory falls below the guard has its container
killed on the first sample below the line, which protects SSH and the host
rather than an unready service. The kill is logged to the journal (tag
`spark-memguard`) with the sample, the line and the shortfall, and with the
previous sample, how long before it was taken and how far MemAvailable fell
since.

After the kill the guard saves what `docker rm -f` would otherwise destroy to
`/var/log/spark-memguard/<container>-<UTC time>/` (`MEMGUARD_EVIDENCE_DIR`):
`meminfo.txt` (`/proc/meminfo` read just before the kill), `ps-rss.txt` (the
largest host processes), `inspect.json` and `container.log` (the last 2000
lines). The journal names that directory. Saving never delays the kill.

A host may set a warn line above the steady guard, `host.memguard_warn_gib`
(default 0: none). The steady guard then logs `low headroom on <container>
during steady` once each time MemAvailable falls below that line, and again
only after it has recovered above it. The warn line only reports; it never
delays or replaces the kill. A fall from above the warn line to below the kill
line within one interval kills without a warning; the kill message then says
so (`no warning: it fell from above the warn line to below the kill line
between two samples`), so a missing warning reads as a fast drop, not a broken
guard.

`bin/spark bench` also refuses to measure a cluster whose live configuration
differs from the selected profile.
