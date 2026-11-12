# Fix `direct_io` External DMA-BUF Device Range Check Under CUDA 13.5 Forward Compatibility

## Root Cause of `candidate-55` `tp3-switched` Boot Failure

On `tp3-switched` (`sp5..sp7`, `results/private/failed-starts/20261112T000453Z-spark-ds41f`), `spark-ds41f:candidate-55` failed simultaneously on all three TP ranks during `finish_weight_transfers() -> DirectWeightSession.flush()` (`vllm/model_executor/model_loader/direct_io/_checkpoint.py:233` / `_batch.c:155`) with:

```text
RuntimeError: batch destination exceeds CUDA device allocation
```

### Causal Chain (Confirmed on `sp8`)

1. **Why only `tp3-switched` hit it (and not `tp4-ring` or `tp4-switched`):**
   - `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0` on `tp4-ring` and `tp4-switched`, whereas `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1` on TP3 (`tp3-triangle`, `tp3-switched`).
   - On TP3, `display_carveout.allocate()` (`vllm/v1/worker/display_carveout.py`) allocates `language_model.model.embed_tokens.weight` (`(43136, 5120)` BF16 = `441,712,640` bytes inside a `432 MiB` = `452,984,832`-byte DRM scanout dumb buffer on `/dev/dri/card0`), `language_model.lm_head.weight`, and DSpark's `model.markov_head.markov_w1.weight` via `cuImportExternalMemory` + `cuExternalMemoryGetMappedBuffer`.
   - Note that `#1134`'s `0800` (`bind_weight_arena`) skips non-resizable `__cuda_array_interface__` storages (`param.untyped_storage()._resizable() == False`) and never touches carve-out weights.

2. **Why `candidate-54` booted on `tp3-switched` while `candidate-55` failed:**
   - `candidate-55` adds `sources.cuda_toolkit = 13.5.37` (`requirements/cuda-13-5-sbsa.txt`, `#974`), which installs `cuda-compat-13-5=620.70.06-1ubuntu1` at `/usr/local/cuda-13.5/compat/libcuda.so.620.70.06` (required on host driver `580.178.04`, which otherwise rejects `ptxas 13.5.37` `sm_121a` cubins in `cuModuleLoad` with `CUDA_ERROR_INVALID_IMAGE = 201`).
   - Under `docker run --gpus all`, `nvidia-container-cli` automatically activates `/usr/local/cuda/compat/libcuda.so.1` (`cuDriverGetVersion() == 13050`).
   - Querying the imported external dma-buf pointer (`0xfbccc0000000`, descriptor `423`, `language_model.model.embed_tokens.weight`):
     - **Host `libcuda.so.580.178.04` (`13000`, `candidate-54`):**
       - `CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2` (`CU_MEMORYTYPE_DEVICE`)
       - `cuMemGetAddressRange(&base, &size, ptr)` -> `base = 0xfbccc0000000, size = 452984832`
     - **Compat `libcuda.so.620.70.06` (`13050`, `candidate-55`):**
       - `CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 1` (`CU_MEMORYTYPE_HOST`), even though `CU_POINTER_ATTRIBUTE_HOST_POINTER` fails with `CUDA_ERROR_INVALID_VALUE (1)` and `CU_POINTER_ATTRIBUTE_DEVICE_POINTER` returns `0xfbccc0000000`
       - `cuMemGetAddressRange(&base, &size, ptr)` returns `CUDA_SUCCESS` with **`base = 0x0`** and `size = 452984832`, while `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` and `CU_POINTER_ATTRIBUTE_RANGE_SIZE` return the exact `0xfbccc0000000` base and `452984832` size (and `cudaMemcpyAsync` H2D + GPU kernels work identically).
   - Both `kind != CU_MEMORYTYPE_DEVICE` and `base == 0x0` caused `direct_io/_cuda_range.h:device_range()` to return `false` at descriptor `423`.

## Fix (`vllm/0804-fix-direct_io-accept-external-dma-buf-device-mapping.patch`)

- In `vllm/model_executor/model_loader/direct_io/_cuda_range.h`:
  - Accept `CU_MEMORYTYPE_HOST` only when `CU_POINTER_ATTRIBUTE_HOST_POINTER != CUDA_SUCCESS` (rejecting all pinned/registered host memory) and `CU_POINTER_ATTRIBUTE_DEVICE_POINTER == pointer`.
  - When `cuMemGetAddressRange` reports `base == 0` on such an external device-only mapping, resolve `base` from `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` after verifying `CU_POINTER_ATTRIBUTE_RANGE_SIZE == size`.
- Applies cleanly with `git am --committer-date-is-author-date` both directly onto `candidate-54` (`15833ed19c5d` -> `9f74b397465f`, tree `1f0513878309`) and directly after `0800..0803` in `candidate-55` (`3987ee2fb302`, tree `e8c20f549a19`).

## Verification on `sp8` (`GB10`, `7.0.0-1024-nvidia`, Driver `580.178.04` + `cuda-compat-13-5` `620.70.06`)

1. `pytest tests/model_executor/model_loader/test_direct_loader.py tests/model_executor/model_loader/test_direct_loader_cuda.py tests/v1/worker/test_display_carveout.py`:
   - Host `libcuda.so.580.178.04` (`cuDriverGetVersion = 13000`): **106 passed in 13.06s** (including `test_external_dmabuf_carveout_loads_and_rejects_out_of_bounds`).
   - Compat `libcuda.so.620.70.06` (`cuDriverGetVersion = 13050`): **106 passed in 17.88s**.
2. Full 40-layer DS4.1 + 3-layer DSpark TP3 `GPUModelRunner.load_model()` under `cuda-compat-13-5` (`cuDriverGetVersion = 13050`), `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`, `--memory=112g`:
   - `Direct loader: weights loaded, 100.57 GB of checkpoint payload in 18.11 s (5.55 GB/s)`
   - `Display carve-out holds 695.0 MiB of streamed weights in 3 buffers, loaded in place (language_model.model.embed_tokens.weight, language_model.lm_head.weight, model.markov_head.markov_w1.weight)`
   - `Model loading took 92.62 GiB memory`; minimum `MemAvailable` across entire load = `12.59 GiB`.
