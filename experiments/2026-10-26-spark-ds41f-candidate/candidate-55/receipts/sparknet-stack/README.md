# candidate-55's sparknet series, 2026-11-11

Built on sp8 (`~/spark-lab/agents/candidate-55-sparknet/dgx-spark-networking`, branch `candidate-55`) from
sparknet `1fd6a1fb`. Every GPU command ran under `flock /tmp/spark-lab-gpu.lock` in a throwaway
`spark-ds41f:candidate-54` container.

| Patch | Source | Commit (sp8) | Tree |
| --- | --- | --- | --- |
| base | sparknet `1fd6a1fb` | `1fd6a1f` | `8cc28a5c` |
| [0001](../../sparknet/0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch) | #1134 (= #1008's sparknet patch), byte-identical | `d3d5b1f` | `dab8c457` |
| [0002](../../sparknet/0002-oneshot-add-single-kernel-multi-stage-pipelined-RoCE.patch) | #989 (= dgx-spark-networking#7), `git am -3`, no conflict | `85a1ea1` | `d8010d07` |
| [0003](../../sparknet/0003-perf-oneshot-chain-the-one-shot-collectives-with-pro.patch) | #982 Part A (= dgx-spark-networking#6), re-placed by hand | `f713f74` | `e06f6a06` |

The exported series replays with `git am` on a fresh tree at `1fd6a1fb` to the same three trees, and
`git diff --check` is clean.

## 0002 on 0001

`roce_all_reduce.py` is the only shared file: 0001 adds the `SPARKNET_ROCE_PROXY_CPU=fast` default in `__init__`, 0002
edits the `should_*` methods and `all_gather` further down, so 0002's hunks only move by five lines. 0002 matches
dgx-spark-networking#7 (`2e2e502`).

## 0003: where the wait and the trigger sit

| Kernel | `T.pdl_sync()` | First global read | `T.pdl_trigger()` |
| --- | --- | --- | --- |
| `_oneshot_tilelang.py` (all-reduce) | :111 | :116 `counters[0]` | :150, after the doorbell (as #982) |
| `_allgather_tilelang.py` | :70 | :73 | :131, under `if s == num_stages - 1:` in the stage loop |
| `_reducescatter_tilelang.py` | :90 | :94 | :156, the same guard |
| `_exchange_tilelang.py` | :86 | :90 | :162, the same guard |

- **The trigger fires once a launch, after the last stage's doorbell.** Every captured (decode) launch is
  single-stage: the runtime pipelines only outside graph capture (`runtime.py` :1298, :1322, :1483, :1617, :1720). So
  a captured launch behaves exactly as #982 measured. Triggering at stage 0 would keep dependent CTAs resident through
  every later stage, on the SMs #989 leaves to the main stream.
- **The trigger's position cannot make a waiting consumer read early.** A dependent's `griddepcontrol.wait` returns
  only after the whole primary grid has completed and flushed. Measured on one GB10, a consumer that loads *before*
  its wait sees exactly the last stage stale (reduce-scatter 4 stages: 8,192 of 32,768 words), which is why every
  consumer must wait first.
- **Part B is left out:** `next_send_buffer` and `_CudaSlotView`, `roce_same_ptr`, the same-pointer guards in
  `roce_copy_pack` and `roce_stage_gather_pack`, and their two tests. No vLLM code calls `next_send_buffer`.

## Correctness review

- **Kernel entry:** in the generated CUDA (TileLang 0.1.15, world 4), only zero-initialised locals precede
  `cudaGridDependencySynchronize()` in all four kernels; the next statement reads `counters[0]`.
- **vLLM's PDL audit** (`deepseek_v41_pdl_sources.violations()`, candidate-54's, with the four kernels in `CHAIN` as
  `0803` adds them): 0 violations on all 18 builds compiled (world 1, 2, 4, 4-ring4, and the traced all-reduce). In a
  captured producer → collective → consumer graph both edges are programmatic for all four kernels.
- **Consumers in candidate-54's vLLM** that follow the collectives wait before reading their output: the WO-B and
  MoE-combine all-reduces feed `mhc_project_streams` (`tilelang/mhc.py:152`, wait first), Engram's
  `engram_post_gate` (`tilelang/engram.py:376/530`), `mhc_post` (`mhc.py:567/616`; reads only `post`, `comb` and
  `residual` before the wait) and the last layer's collapse kernels; the fused WQA/WKV all-gather feeds
  `attention_front` (`tilelang/front.py:285`; reads only weights, slots, positions and `cos_sin` before the wait); the
  compressor all-gather feeds `gather_columns_into` (`tilelang/gather.py:69`). Kernels without PDL are fully
  serialized after the collective. None reads a collective's output, or its producer's, before its wait.
- **The two-slot ping-pong** is unchanged: stage s+1 stages into `send[(epoch+s+1)&1]` only after
  `roce_wait_peers(epoch+s)`, and the next launch reads `counters[0]` only after this grid completes.

## Tests

| Test | Result |
| --- | --- |
| CPU suite (`python3 -m unittest discover -s tests`) after each patch | 97, 98, 99 OK (the ASan/UBSan proxy simulator and the PDL source test included) |
| Proxy simulator, final tree (`-fsanitize=address,undefined`) | PASS: direct3/ring4/mesh4, 1/2/4 paths, 31,000 collectives |
| pytest in the candidate-54 container | 99 passed, 102 skipped (`tests/gpu/test_oneshot_gpu.py` needs `torchrun` with `WORLD_SIZE >= 2`) ([pytest.log](pytest.log)) |
| [pdl_check.py](pdl_check.py) 20 replays, one GB10 | 14 functional cases exact (all four collectives at world 1 with 1 and 4 stages, world 2 rank 0 with the peer's slots pre-filled, every buffer poisoned before each replay); 3 early-reader controls stale as expected; "PDL CHECK OK" ([pdl_check.log](pdl_check.log), [pdl_check.json](pdl_check.json)) |

**Not run:** the multi-rank RoCE suite and any multi-node benchmark. #989's and #982's own marks run on sp1..sp4 on
candidate-55's image.
