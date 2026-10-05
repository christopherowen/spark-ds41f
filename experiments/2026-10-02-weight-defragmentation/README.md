# Checkpoint defragmentation (paused)

Base deployment commit: `4852c9a` (r5o, image `sha256:288fc5bd`), checkpoint
revision `dba1be0a`. The run was owner-authorized and production stayed up.

`doctor --live` now reports checkpoint shards above ext4's best extent count
(`scripts/weight_fragmentation.py`, `sudo -n e4defrag -c`). `bin/spark3 doctor
--fragmentation-commands` prints `e4defrag` commands for the affected blobs and
never runs them. This experiment tried those corrections on the 48 shards per
node while serving continued.

## Method

[defragment.py](defragment.py) ran per node in a bounded systemd unit:
- **Limits:** memory 1 GiB with no swap, nice 19, idle I/O class, one CPU.
- **Abort rule:** stop if MemAvailable fell below 4 GiB.
- **Per shard:**
  1. checksum the resolved blob with direct-read SHA-256 against its
     content-addressed name;
  2. run `e4defrag -v`;
  3. checksum it again.

## Result (2026-10-02 04:28-04:31 UTC)

Paused, awaiting a serving stop or restart.

- **Memory limit:** the 1 GiB cgroup ran out of memory partway through:
  dgx2 at shard 3, dgx3 at shard 6. dgx1 was stopped at shard 7.
- **Data intact:** every shard the run touched, interrupted ones included,
  then passed the full SHA-256 check against its original blob name.
- **What changed:** where `e4defrag` ran, some shards reached their best count
  (dgx1 shard 2: 33 extents to 2). Others did not change.
- **Not established:** that fragmentation slows loading. Boot reads the 101 GB
  checkpoint in about 15 s.

Per-node logs are in [runs/](runs/) and the summary is in
[results.json](results.json). The first dgx3 canary stopped before touching a
file because root's git rejected the checkout's ownership. A command-local
`safe.directory` fixed that.

Resume only in a maintenance window with serving stopped, and with a larger
memory limit.
