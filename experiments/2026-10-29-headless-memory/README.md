# What the idle headless launcher holds

On every node but the head, `vllm serve --headless` starts the node's worker and then only
monitors it, for the life of the deployment. The TP3 memory study's soak
(`../2026-10-26-tp3-budget-memory`, "The rank-0 tax") measured it at 0.98 GiB of anonymous
memory, flat. On GB10 that is MemAvailable the guards and the long prefills share.

## Candidate 12's launcher on dgx2 (2026-10-29, read from /proc)

During candidate 12's qualification at TP4 (pid 73395, the production argv and environment):

- Pss_Anon 1,003 MiB, all private dirty: the brk heap 371 MiB, 78 single mappings above 1 MiB
  totalling 488 MiB (with `MALLOC_MMAP_THRESHOLD_=1048576`, each allocation of 1 MiB or more
  is its own mapping; sizes 3 to 84 MiB, none repeated), 87 small arenas 25 MiB. Not glibc's
  thread arenas: `MALLOC_ARENA_MAX=2`, and the 17 64 MiB regions hold 1 MiB between them.
- `libcuda`, `libtorch_cuda` and `libnccl` are loaded, and `/dev/nvidiactl`, `/dev/nvidia0`
  and `/dev/nvidia-uvm` are open (13 descriptors): the launcher initialised the driver, though
  `nvidia-smi` lists only the worker as a compute process.
- 26 threads.

## The probe (`launcher.json`, run `headless-launcher-memory`)

[bundles/launcher](bundles/launcher): the launcher's argv (`argv.json`, from the running
process) through the real `vllm` CLI in candidate 12's image on dgx4, with `run_headless`
replaced by its own steps up to the executor (the executor would spawn a worker and join the
cluster). After each stage (python, torch, the vLLM CLI's imports, the parsed arguments, the
engine args, the engine config, the executor's import, CUDA's device count and properties) it
records anonymous memory by mapping kind and whether the driver is open, then the same after
`gc.collect()` and `malloc_trim(0)`.

What the answer decides:
- **Garbage** (gc and trim give it back): the launcher trims once its worker is up.
- **Imports and the config:** the launcher cannot shed them in-process; the worker would have
  to run without it (the node's process is the worker), a larger vLLM change.
- **The driver:** the launcher stops initialising CUDA (the worker spawns either way).

## Results

Measured on `dgx4` inside `spark-ds41f:1030` (`launcher_memory.py` with the production argv):

| Stage | `Pss_Anon` | `brk` heap | Single mappings $\ge 1\text{ MiB}$ | Small arenas | CUDA driver open |
|---|---:|---:|---:|---:|---|
| `python` | `9 MiB` | `1 MiB` | `3 MiB` | `5 MiB` | No |
| `import torch` | `303 MiB` | `52 MiB` | `124 MiB` | `126 MiB` | No |
| `import vllm cli` | `660 MiB` | `204 MiB` | `258 MiB` | `197 MiB` | No |
| `args parsed` / `engine args` | `736 MiB` | `252 MiB` | `278 MiB` | `204 MiB` | No |
| `engine config` | `1,001 MiB` | `395 MiB` | `399 MiB` | `206 MiB` | Yes (`13` fds) |
| `engine config + gc/trim` | `966 MiB` | `392 MiB` | `369 MiB` | `204 MiB` | Yes (`13` fds) |

Conclusion:
- `gc.collect()` + `malloc_trim(0)` reclaims `35 MiB`, while the remaining `966 MiB` (`~0.95 GiB`) is live Python module and config state (`torch`, `vllm`, `transformers`, and `vllm_config`) plus CUDA driver initialization.
- On single-rank follower nodes (`node_rank_within_dp > 0`, `local_world_size == 1`, `monitor_workers=False`), `MultiprocExecutor` creates no local `rpc_broadcast_mq` or `response_mqs`; running `WorkerProc.worker_main` inline in the `vllm serve --headless` process (`0059-headless-inline-worker-and-heap-trim.patch`) eliminates the duplicate `~1.00 GiB` parent process on `dgx2`, `dgx3`, and `dgx4` while trimming `~35 MiB` of glibc heap slack after startup across all nodes.
