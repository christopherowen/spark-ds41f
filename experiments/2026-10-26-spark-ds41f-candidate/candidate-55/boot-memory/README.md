# Boot memory at tp3-switched: sixth against seventh build

The seventh build's lowest MemAvailable during a boot on sp5 is 5.0 to 5.1 GiB, against 6.3 to 6.7 GiB for builds five
and six, and the engine's breakdown gives 15.05 GiB outside torch's allocator against 13.78 GiB (fourth build).

One boot of each build, no benchmark, to read from the running engine on sp5, sp6 and sp7: the `memory_breakdown`
line, and each engine process's resident, anonymous, shared and file-backed memory.

- `tp3-switched-b6.json`: the sixth build's image (`spark-ds41f:candidate-55-b6`) with its lock.
- `boot-b6-tp3-switched.json`, `boot-b7-tp3-switched.json`: one boot each.

The lock here is the sixth build's with its own series file (`vllm-series`, the 680 patches before `0808`), so its
patch-set fingerprint is this file's; the patch head and tree are the sixth build's.

## Result (2026-11-14, sp5, rank 0, about two minutes after ready)

| | sixth build | seventh build |
| --- | ---: | ---: |
| MemAvailable | 7.01 GiB | 5.59 GiB |
| Engine breakdown: outside torch's allocator | 13.80 GiB | 15.07 GiB |
| Worker: resident / shared / anonymous | 4.49 / 0.49 / 3.10 GiB | 5.56 / 1.32 / 3.31 GiB |
| Worker: `/dev/shm/nccl-*` segments | 2 of 33 MiB | 10 of 65 MiB |
| Worker: `/dev/zero` shared mappings | 432 MiB | 696 MiB |
| Worker threads | 91 | 107 |
| API server, engine core (resident) | 1.48, 1.39 GiB | 1.50, 1.41 GiB |

The difference is NCCL communicators, not torch's libraries: one process importing torch, creating a CUDA context,
running cuBLAS and one NCCL communicator takes the same memory in both images within 0.1 GiB.

**Cause, from torch's source.** torch 2.16 registers the `nccl` backend as `ProcessGroupNCCL2` unless
`TORCH_DIST_USE_NCCL2=0` (`torch/distributed/distributed_c10d.py`, `_register_builtin_nccl_backend`). That
implementation binds a device and creates the NCCL communicator when a process group is constructed; the classic
`ProcessGroupNCCL` creates it on the group's first collective. vLLM constructs a device group for each of its
coordinators, so every rank holds a communicator, its proxy threads, its shared pool and its device buffers for
groups that never run a collective.

`tp3-switched-classic-groups.json` sets `TORCH_DIST_USE_NCCL2=0` on the seventh build;
`boot-b7-classic-groups-tp3-switched.json` boots it once.

**With `TORCH_DIST_USE_NCCL2=0`** (one boot, sp5, rank 0): two `nccl-*` segments, `/dev/zero` mappings 432 MiB and 91
threads, as on the sixth build; MemAvailable 6.49 GiB and 14.04 GiB outside torch's allocator. That recovers 0.90 of
the 1.42 GiB. Of the 0.52 GiB left against the sixth build, 0.32 GiB is torch's allocator holding more than it has
allocated (0.73 against 0.41 GiB) and 0.06 GiB the two pools' larger size in NCCL 2.32.3; the rest is not attributed.
The setting is in candidate-55's shape profiles from here; it does nothing on a torch that lacks the variable.
