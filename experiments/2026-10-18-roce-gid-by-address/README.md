# RoCE GIDs by address: r6v with no GID slot for NCCL or sparknet

**Plan, not run.** Nothing here has touched the cluster. The window it needs
reboots one node while the candidate serves. That is the owner's call, like
every start, stop and restart (AGENTS.md).

## Why

On 2026-10-11 dgx1 rebooted. Afterwards dgx2's and dgx4's dgx1-facing ports had
GID slot 3 empty and the IPv4 RoCE v2 GID in slot 4. The links had flapped while
running ranks still referenced the old entry, and the kernel files a re-added
address in the lowest *free* slot. Production pins both libraries to slot 3:
`NCCL_IB_GID_INDEX=3` in the shapes, and `SPARKNET_ROCE_GID_INDEX=3`, which
sparknet before 0.5.0 reads. So the restart failed until a cluster stop and a
`sudo nmcli connection up` on each affected connection put the address back in 3.
The addresses never changed: they are static netplan, and they change only on
recabling.

The slot is the wrong key. NCCL 2.30.7 finds the GID itself when
`NCCL_IB_GID_INDEX` is unset (`ncclIbGetGidIndex` in
`src/transport/net_ib/connect.cc` scans the table by `NCCL_IB_ADDR_FAMILY`,
`NCCL_IB_ADDR_RANGE` and `NCCL_IB_ROCE_VERSION_NUM`, and the shapes already set
`AF_INET` and `2`). librdmacm selects by source address too. sparknet 0.5.0
([christopherowen/dgx-spark-networking#1](https://github.com/christopherowen/dgx-spark-networking/pull/1))
does the same, strictly. When it opens each HCA it scans the port's GID table with
`ibv_query_gid_ex` and takes the one entry that is RoCE v2, IPv4-mapped and inside
that HCA's `SPARKNET_ROCE_SUBNETS` network. None, or more than one, fails setup
naming the device; there is no fallback to a slot.

## Intended delta

One variable: how the source GID is chosen, for both libraries.

| | r6v (production) | candidate |
| --- | --- | --- |
| sparknet | `3bc8d10` (tree `f72599fa`) | `5f335aa` (tree `dae2ca2f`), branch `roce-gid-discovery`: `3bc8d10` + 0.4.0 (docs and the version only) + the GID change |
| NCCL | `NCCL_IB_GID_INDEX=3` | unset: NCCL scans (`NCCL_IB_ADDR_FAMILY=AF_INET`, `NCCL_IB_ROCE_VERSION_NUM=2`) |
| sparknet's slot | `SPARKNET_ROCE_GID_INDEX=3` | unset (0.5.0 refuses it); `cluster start` renders each node's `SPARKNET_ROCE_SUBNETS` from `roce_subnets` |
| everything else | r6v | r6v: vLLM tree `d11e9836`, B12X, NCCL `6af10aa7`, TileLang, TileKernels, the 1017 weights |

Between `3bc8d10` and `5f335aa` the kernels, the proxy's posting logic and wire
ABI 10 are unchanged. The proxy's setup takes per-HCA subnets instead of an
index, and the RTR address vector uses each HCA's own slot. The runtime's
API version is 3 (the `gid_index` keyword is gone); the vLLM adapter in the same
package requires 3, and vLLM 0038 passes no index.

Files:

- `upstreams.lock.json`, `source.json`: r6v's, with sparknet moved.
- [`gid-tp4.json`](gid-tp4.json), [`gid-tp3.json`](gid-tp3.json): the recipe at
  each shape with the image `vllm-ds41f-kkref:19f2c20ed4d6-r6v-gid` and both pins
  unset. [`gid-tp4-nccl-info.json`](gid-tp4-nccl-info.json) adds
  `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET`, so each NCCL connection
  logs the GID index it took (`NET/IB: ... GID <index> (...)`).
- [`gid_check.py`](gid_check.py): a fabric script. On every rank it builds
  sparknet's runtime from the arm's environment and compares the slot the
  proxy's verbs scan took (`stats()["gid_indices"]`) with sparknet's rule over
  the sysfs table, then checks one all-reduce.

## Hypothesis and acceptance

1. **Healthy fabric (every port's GID at 3):** both libraries take slot 3, so
   nothing measurable changes. The temperature-0 and greedy texts are
   byte-identical to r6v's gate (`3eab5112` / `db25fde3`, greedy `c9cf9dbe`,
   [r6v](../2026-10-17-r6v/README.md)). 16, 8 and 4 distinct streams are within
   r6v's noise of 243.4, 184.2 and 137.0 tok/s, and the boot is ready in r6v's
   band (96 s). Any text difference fails the candidate. It would mean a
   library took another GID or the build differs.
2. **After a neighbour reboot under live RDMA:** the moved ports' GIDs sit at
   index 4, and the candidate restarts as it is: no `nmcli`, no doctor finding
   about slots. sparknet's runtime reports slot 4 for those HCAs, NCCL logs GID
   4 on them, and the texts are identical to (1). Production's pinned
   configuration reports the same ports as `GID index 3 (fixed by
   SPARKNET_ROCE_GID_INDEX, NCCL_IB_GID_INDEX) is empty`, which shows that the
   check reproduced the incident.
3. **Fail-stop and probe:** the collective probe and both of sparknet's fail-stop
   scenarios pass on all four ranks, as for r6n.

Rollback: production's configuration is untouched (`config/cluster.json` still
pins slot 3), so closing the window restores r6v. Re-activate the moved
connections first (step 6 below), or production cannot start.

## Before the window (no effect on serving)

1. Merge the spark-ds41f pull request that carries this directory: a window opens
   only at a commit on `origin/main`. It changes no rendered value of the
   production configuration, which pins both slots. The doctor's live check gains
   the by-address rule beside its slot-3 check.
2. Delete `"roce_gid_index": 3` from the site node maps on dgx1
   (`config/nodes.json`, `config/nodes-ring4.local.json` and the other
   `*.local.json` maps). `cluster sync` copies the selected map to the nodes.
   Production renders the same environment without it (checked locally on all
   four maps); the candidate arms refuse a map that still carries it.
3. Merge sparknet's PR, or leave the lock on the branch commit. The build checks
   out `5f335aa` either way.

## Runs (TP4 window)

The site is cabled as the ring. TP3 waits for the triangle (below).

1. **Image** ([build.json](build.json)): built on dgx2 and copied, as r6v was.
   Confirm the label `local.spark3.sparknet.tree=dae2ca2f...` and, in the image,
   `python3 -c 'import sparknet; print(sparknet.__version__, sparknet.API_VERSION)'`
   prints `0.5.0 3`.
2. **Qualify** ([qualify-tp4.json](qualify-tp4.json), serving stopped):
   `gid_check.py` on every rank (the proxy's slot equals the sysfs rule's, and
   the sum is right), then the collective probe and the fail-stop scenarios
   (eager, graph).
3. **Gate** ([gate-tp4.json](gate-tp4.json)): r6v's gate on the candidate:
   determinism and greedy text, then 16, 8 and 4 distinct streams. Record the
   boot's ready time.
4. **Link flap**, by hand in the same window, after the gate. Keep the window's
   heartbeat fresh on dgx1 while working by hand:
   `while sleep 60; do python3 -c 'import sys; sys.path.insert(0, "scripts"); import lab; lab.beat("gid flap check")'; done`.
   Reboot **dgx3**, not dgx1. dgx1 holds the window, its heartbeat and the
   watchdog, and dgx3's ring neighbours (dgx2, dgx4) exercise the same path.
   1. Boot the NCCL-logging arm and serve one request:
      `bin/spark --cluster-config experiments/2026-10-18-roce-gid-by-address/gid-tp4-nccl-info.json cluster start --apply --replace`.
      Save `bin/spark ... doctor --live` (expect no GID finding), and
      `ibv_devinfo -v | grep 'GID\['` on dgx2 and dgx4.
   2. With the ranks up and holding their queue pairs, reboot dgx3:
      `ssh dgx3 sudo -n reboot`. Expect the serving group to fail-stop
      (`saw nothing from rank 2`, or NCCL's error). That is the incident, not a
      finding.
   3. When dgx3 is back and its links are up, save `ibv_devinfo -v | grep 'GID\['`
      on dgx2 and dgx4 again. **The check counts only if a dgx3-facing port's
      IPv4 RoCE v2 GID moved** (slot 3 empty, the address at 4). If none moved,
      the ranks released their references before the links returned; record it
      and repeat from step 1 rather than call it a pass.
   4. Without any `nmcli`: run `doctor --live` with production's configuration
      (expect `GID index 3 (fixed by ...) is empty` on the moved ports, the
      control) and with `gid-tp4-nccl-info.json` (expect only the stopped
      service). Then `cluster stop --apply` and `cluster start --apply --replace`
      with `gid-tp4-nccl-info.json`. It must come up and serve. In each moved
      node's container log, NCCL's `NET/IB` lines for the moved devices name
      GID 4.
   5. Stop serving, then run [qualify-tp4.json](qualify-tp4.json)'s `gid_check`
      job (a one-job spec, or `queue add`): it must report slot 4 for the moved
      HCAs on dgx2 and dgx4, agreeing with sysfs. Boot `gid-tp4.json` and run the
      gate's determinism extra: texts identical to step 3.
   6. **Restore the slots before closing**: stop serving (nothing may hold RDMA),
      then re-activate each moved connection: on dgx2
      `sudo -n nmcli connection up "cx7 dgx2-dgx3 path1"` and `... path2`, on
      dgx4 `"cx7 dgx3-dgx4 path1"` and `... path2`. Check that
      `/sys/class/infiniband/<device>/ports/1/gid_attrs/types/3` is `RoCE v2` on
      all 16 ports, then close the window. Production (pinned) restarts on
      slot 3.
5. Write the results here: every run, including the failed or inconclusive
   ones.

## TP3

`gid-tp3.json` is the same change at `tp3-triangle`. When the triangle is
cabled, run the image, qualify and gate jobs at that shape (TP3 has no ring
relay; `qualify` covers `oneshot-direct`). The flap check is the same mechanism
on a different cable plan; one pass on the ring covers it unless TP3 is promoted
first.

## Promotion (owner)

If the gates pass: promote to the recipe in one commit with a new baseline.
`config/cluster.json` drops `NCCL_IB_GID_INDEX` from both shapes and
`SPARKNET_ROCE_GID_INDEX` from the top-level environment, and takes this
directory's image, lock and source manifest. `tests/test_recipe.py` names the
released arms. `docs/setup.md`, `docs/doctor.md` and
`docs/switchless-topology.md` drop "until the by-address release is promoted",
and the slot repair (`nmcli` after a neighbour reboot) no longer applies to the
recipe. The doctor keeps the slot-3 check only for configurations
that still pin a slot: B12X's `rocenante-*` transports and archived releases.

## Out of scope: B12X's RoCEnante

B12X's one-shot path (`rocenante-*`, `patches/b12x`) keeps one fixed slot:
`B12X_ROCE_GID_INDEX`, else `NCCL_IB_GID_INDEX`, else 3. No recipe shape uses
it, and B12X's collectives are being retired in favour of sparknet's. Its
configurations stay "pinned" (`topology.gid_pinned`). The doctor checks their
slot and the address both, and `roce_gid_index` may still move one node's slot
for them. Giving it the same selection would be a new B12X patch (C and Python)
with its own hardware qualification.
