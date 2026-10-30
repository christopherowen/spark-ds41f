# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the b12x project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--load-format direct``: safetensors weights read with O_DIRECT through a registered
host ring and copied straight into the model's CUDA parameters.

The loader is for unified-memory GPUs such as GB10, where the GPU reads pageable
memory through the host page tables. There a load through the page cache costs twice:

- every byte read stays in the page cache, which counts as available memory but takes
  the free memory the CUDA allocator needs for the KV cache and graphs allocated after
  loading;
- a copy from a memory-mapped checkpoint page the process has not touched can hang
  (driver R580, 64 KiB pages); ``weight_transfer.copy_weight`` copies such sources into
  anonymous memory first.

So nothing here maps the checkpoint or reads it through the page cache:

- The weights iterator yields meta tensors that carry each tensor's checkpoint range
  (``direct_io.DirectWeightSession``). Model weight loaders route them into parameter
  views as usual, and ``weight_transfer`` hands each copy to the session, which
  records it and later reads it with O_DIRECT through a 16 MiB registered ring into
  its destination. Scalars and the model's declared metadata are read into CPU
  tensors, and copies the session declines go to Torch.
- Tensors the model reads from the checkpoint files while serving
  (``checkpoint_file_weight_filter``, such as DeepSeek-V4.1's Engram tables) never pass
  through the reader: the iterator yields vLLM's file-source descriptors for them
  (``weight_utils.file_source_tensor``), in name order with the file's other tensors.

The device must report pageable memory access through the host page tables; the
loader refuses others, since it has no GPU-direct-storage path. The native helper
(``direct_io/_native.py``) is built on first use. ``model_loader_extra_config`` takes
``io_threads``, the reads in flight (1-16, default 8). The loader logs one summary line
once the weights are loaded.

Ported from local-inference-lab/b12x b12x/integration/vllm/loader.py at 236ddff04a5f
(Apache-2.0) and vLLM's copy of it (``b12x_loader.py``). Changed: no GDS transport,
shared read group, IPC-exportable allocations or progress panel.
"""

import dataclasses
import heapq
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.logger import init_logger
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.direct_io import (
    DirectWeightSession,
    capabilities,
)
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.weight_transfer import weight_transfer
from vllm.v1.worker import display_carveout

logger = init_logger(__name__)

_ARENA_ALIGN_BYTES = 256
_PRESERVE_STORAGE_MAX_BYTES = 1 << 20
_TOUCH_STRIDE_BYTES = 1 << 21
_SPONGE_CHUNK_BYTES = 1 << 27
_SPONGE_HEADROOM_BYTES = 4 << 30
_SPONGE_MIN_MODEL_BYTES = 1 << 30


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def parse_buddyinfo_order13(
    text: str, page_bytes: int | None = None
) -> tuple[int, int]:
    """Return ``(sub_order13_bytes, order13_plus_bytes)`` from ``/proc/buddyinfo``.

    On 64 KiB-page kernels (such as GB10), buddy order 13 is a 512 MiB contiguous
    block. Returns ``(0, 0)`` when ``page_bytes`` is not 64 KiB.
    """
    if page_bytes is None:
        try:
            page_bytes = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            return 0, 0
    if page_bytes != 65536:
        return 0, 0
    sub_order13 = 0
    order13_plus = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) <= 4 or parts[0] != "Node" or parts[2] != "zone":
            continue
        for order, raw in enumerate(parts[4:]):
            try:
                count = int(raw)
            except ValueError:
                continue
            block_bytes = count * (page_bytes << order)
            if order < 13:
                sub_order13 += block_bytes
            else:
                order13_plus += block_bytes
    return sub_order13, order13_plus


def _allocate_order13_sponge(
    device: torch.device,
    *,
    required_bytes: int,
    buddyinfo_path: Path = Path("/proc/buddyinfo"),
    page_bytes: int | None = None,
    headroom_bytes: int = _SPONGE_HEADROOM_BYTES,
    chunk_bytes: int = _SPONGE_CHUNK_BYTES,
) -> list[torch.Tensor]:
    """Temporarily soak ``< 512 MiB`` buddy blocks so subsequent model weight
    allocations draw from ``Order 13`` (``512 MiB``) blocks first."""
    try:
        text = buddyinfo_path.read_text()
    except OSError:
        return []
    sub_order13_bytes, order13_bytes = parse_buddyinfo_order13(
        text, page_bytes=page_bytes
    )
    if sub_order13_bytes < chunk_bytes or order13_bytes < chunk_bytes:
        return []
    try:
        free_gpu_bytes, _ = torch.accelerator.get_memory_info(device)
    except Exception:
        return []
    budget = min(sub_order13_bytes, free_gpu_bytes - required_bytes - headroom_bytes)
    num_chunks = budget // chunk_bytes
    if num_chunks <= 0:
        return []
    sponge: list[torch.Tensor] = []
    try:
        for _ in range(int(num_chunks)):
            chunk = torch.empty((chunk_bytes,), dtype=torch.uint8, device=device)
            if chunk_bytes >= _TOUCH_STRIDE_BYTES:
                chunk[::_TOUCH_STRIDE_BYTES].zero_()
            elif chunk_bytes > 0:
                chunk[:1].zero_()
            sponge.append(chunk)
        if sponge:
            torch.accelerator.synchronize(device)
    except torch.OutOfMemoryError:
        pass
    return sponge


def bind_weight_arena(
    model: torch.nn.Module,
    device: torch.device,
    *,
    order13_sponge: bool = True,
    sponge_min_model_bytes: int = _SPONGE_MIN_MODEL_BYTES,
    buddyinfo_path: Path = Path("/proc/buddyinfo"),
    page_bytes: int | None = None,
) -> torch.Tensor | None:
    """Pack persistent CUDA model parameters into a 256-byte-aligned contiguous
    ``uint8`` arena (``model._direct_weight_arena``) before ``load_weights``.

    Parameters on ``FusedMoEMethodBase`` modules (which ``process_weights_after_loading``
    replaces layer by layer via ``replace_parameter``) keep individual storages so their
    pre-tile weights can be freed during post-processing, but are re-allocated largest
    first under the same Order-13 buddy sponge. Display carve-out tensors and file-backed
    ``load_view`` parameters are left untouched.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    moe_storage_ptrs: set[int] = set()
    for module in model.modules():
        if isinstance(getattr(module, "quant_method", None), FusedMoEMethodBase):
            for param in module.parameters(recurse=True):
                data = param.data
                if (
                    data.device.type == "cuda"
                    and data.numel() > 0
                    and data.untyped_storage().nbytes() > 0
                ):
                    moe_storage_ptrs.add(data.untyped_storage().data_ptr())

    arena_groups: dict[
        int,
        tuple[
            int,
            list[
                tuple[
                    torch.nn.Parameter, int, int, tuple[int, ...], tuple[int, ...]
                ]
            ],
        ],
    ] = {}
    moe_groups: dict[
        int,
        tuple[
            int,
            list[
                tuple[
                    torch.nn.Parameter, int, int, tuple[int, ...], tuple[int, ...]
                ]
            ],
        ],
    ] = {}
    for param in model.parameters():
        if hasattr(param, "load_view"):
            continue
        data = param.data
        if data.device.type != "cuda" or data.numel() == 0:
            continue
        if any(display_carveout._holds(c, data) for c in display_carveout._carveouts):
            continue
        storage = data.untyped_storage()
        nbytes = storage.nbytes()
        if nbytes == 0:
            continue
        ptr = storage.data_ptr()
        target = moe_groups if ptr in moe_storage_ptrs else arena_groups
        if ptr not in target:
            target[ptr] = (nbytes, [])
        target[ptr][1].append(
            (
                param,
                data.element_size(),
                data.storage_offset(),
                tuple(data.shape),
                tuple(data.stride()),
            )
        )

    if not arena_groups and not moe_groups:
        return None

    arena_bytes = sum(
        _align_up(nbytes, _ARENA_ALIGN_BYTES) for nbytes, _ in arena_groups.values()
    )
    moe_bytes = sum(nbytes for nbytes, _ in moe_groups.values())
    total_bytes = arena_bytes + moe_bytes

    cpu_snapshots: dict[int, torch.Tensor] = {}
    empty_storage = torch.empty((0,), dtype=torch.uint8, device=device).untyped_storage()
    with torch.no_grad():
        for groups in (arena_groups, moe_groups):
            for ptr, (nbytes, views) in groups.items():
                if nbytes <= _PRESERVE_STORAGE_MAX_BYTES:
                    orig_storage = views[0][0].data.untyped_storage()
                    src = torch.empty((0,), dtype=torch.uint8, device=device).set_(
                        orig_storage
                    )
                    cpu_copy = torch.empty((nbytes,), dtype=torch.uint8, device="cpu")
                    cpu_copy.copy_(src)
                    cpu_snapshots[ptr] = cpu_copy
                    del src, orig_storage
                for param, _, _, _, _ in views:
                    param.set_(empty_storage, 0, (0,), (1,))
    torch.accelerator.empty_cache()

    sponge = (
        _allocate_order13_sponge(
            device,
            required_bytes=total_bytes,
            buddyinfo_path=buddyinfo_path,
            page_bytes=page_bytes,
        )
        if order13_sponge and total_bytes >= sponge_min_model_bytes
        else []
    )
    arena: torch.Tensor | None = None
    try:
        with torch.no_grad():
            if arena_bytes > 0:
                arena = torch.empty((arena_bytes,), dtype=torch.uint8, device=device)
                if arena_bytes >= _TOUCH_STRIDE_BYTES:
                    arena[::_TOUCH_STRIDE_BYTES].zero_()
                arena_storage = arena.untyped_storage()
                offset = 0
                for ptr, (nbytes, views) in arena_groups.items():
                    if ptr in cpu_snapshots:
                        arena[offset : offset + nbytes].copy_(cpu_snapshots.pop(ptr))
                    for param, elem_size, orig_offset, shape, stride in views:
                        param.set_(
                            arena_storage,
                            (offset // elem_size) + orig_offset,
                            shape,
                            stride,
                        )
                    offset += _align_up(nbytes, _ARENA_ALIGN_BYTES)
                model._direct_weight_arena = arena

            for ptr, (nbytes, views) in sorted(
                moe_groups.items(), key=lambda item: item[1][0], reverse=True
            ):
                moe_buf = torch.empty((nbytes,), dtype=torch.uint8, device=device)
                if nbytes >= _TOUCH_STRIDE_BYTES:
                    moe_buf[::_TOUCH_STRIDE_BYTES].zero_()
                if ptr in cpu_snapshots:
                    moe_buf.copy_(cpu_snapshots.pop(ptr))
                moe_storage = moe_buf.untyped_storage()
                for param, _, orig_offset, shape, stride in views:
                    param.set_(moe_storage, orig_offset, shape, stride)
                del moe_buf, moe_storage

        if sponge or total_bytes >= _TOUCH_STRIDE_BYTES:
            torch.accelerator.synchronize(device)
    finally:
        if sponge:
            sponge.clear()
            torch.accelerator.empty_cache()

    return arena


class DirectModelLoader(DefaultModelLoader):
    """Route checkpoint views through an asynchronous O_DIRECT read ring."""

    def __init__(self, load_config):
        options = dict(load_config.model_loader_extra_config)
        self.io_threads = options.pop("io_threads", 8)
        self.weight_arena = bool(options.pop("weight_arena", True))
        self.order13_sponge = bool(options.pop("order13_sponge", True))
        if options.get("enable_multithread_load"):
            raise ValueError("the direct loader routes checkpoint shards in one thread")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "the direct loader's O_DIRECT reads use no safetensors read strategy"
            )
        super().__init__(
            dataclasses.replace(
                load_config,
                load_format="safetensors",
                model_loader_extra_config=options,
            )
        )
        self._session = None

    def create_model(self, vllm_config, model_config, prefix=""):
        model = super().create_model(vllm_config, model_config, prefix)
        if self.weight_arena:
            device = torch.device(
                self.load_config.device or vllm_config.device_config.device
            )
            bind_weight_arena(
                model,
                device,
                order13_sponge=self.order13_sponge,
            )
        return model

    def load_model(self, vllm_config, model_config, prefix=""):
        device = torch.device(
            self.load_config.device or vllm_config.device_config.device
        )
        if device.type != "cuda":
            raise ValueError("the direct loader requires a CUDA device")
        index = (
            torch.accelerator.current_device_index()
            if device.index is None
            else device.index
        )
        if not capabilities(index)["host_page_tables"]:
            raise RuntimeError(
                f"--load-format direct needs a GPU that reads pageable memory through "
                f"the host page tables, such as GB10, and CUDA device {index} does "
                f"not; it has no GPU-direct-storage path"
            )
        with (
            DirectWeightSession(index, io_threads=self.io_threads) as session,
            weight_transfer(session),
        ):
            self._session = session
            try:
                model = super().load_model(vllm_config, model_config, prefix)
                stats = session.stats()
            finally:
                self._session = None
        torch.accelerator.empty_cache()
        seconds = (
            self.counter_after_loading_weights - self.counter_before_loading_weights
        )
        payload = stats["payload_bytes"] / 1e9
        logger.info(
            "Direct loader: weights loaded, %.2f GB of checkpoint payload in %.2f s "
            "(%.2f GB/s); %.2f GB read from disk, %.2f GB BF16 expanded in the ring, "
            "%.2f GB converted in the GPU scratch, %.2f GB copied by Torch",
            payload,
            seconds,
            payload / seconds if seconds > 0 else 0.0,
            stats["physical_bytes"] / 1e9,
            stats["bf16_expansion_bytes"] / 1e9,
            stats["transform_bytes"] / 1e9,
            stats["torch_copy_bytes"] / 1e9,
        )
        logger.debug("Direct loader I/O counters: %s", stats)
        return model

    @staticmethod
    def _needs_values(entry):
        return math.prod(entry.shape) == 1 or entry.name.rsplit(".", 1)[-1] in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        }

    def _file_backed_weights_iterator(self, files, source, index_path):
        session = self._session
        if session is None:
            raise RuntimeError("the direct loader reads weights only in load_model")
        weight_map = None
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        indexed_paths = {}
        for path in files:
            sources = weight_utils.safetensors_file_sources(path)
            file_names = {name for name in sources if source.file_weight_filter(name)}
            resolved_path = Path(path).resolve()
            selected = []
            for name in sorted(sources):
                if source.weight_name_prefixes and not name.startswith(
                    source.weight_name_prefixes
                ):
                    continue
                if weight_map is not None:
                    indexed_file = weight_map.get(name)
                    if indexed_file is None:
                        continue
                    if indexed_file not in indexed_paths:
                        indexed_paths[indexed_file] = (
                            index_path.parent / indexed_file
                        ).resolve()
                    if indexed_paths[indexed_file] != resolved_path:
                        continue
                if not should_skip_weight(name, self.local_expert_ids):
                    selected.append(name)
            file_backed = (
                (source.prefix + name, weight_utils.file_source_tensor(sources[name]))
                for name in selected
                if name in file_names
            )
            # File-backed entries never enter the checkpoint direct reader, even
            # for mixed files or before metadata-value reads and routing.
            ordinary = (
                session.weights(
                    [path],
                    prefixes=source.weight_name_prefixes,
                    prefix=source.prefix,
                    index_path=index_path,
                    needs_values=self._needs_values,
                    skip=lambda name, file_names=file_names: (
                        name in file_names
                        or should_skip_weight(name, self.local_expert_ids)
                    ),
                )
                if any(name not in file_names for name in selected)
                else ()
            )
            yield from heapq.merge(file_backed, ordinary, key=lambda item: item[0])

    def _get_weights_iterator(self, source):
        session = self._session
        if session is None:
            raise RuntimeError("the direct loader reads weights only in load_model")
        folder, files, _ = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            False,
            source.allow_patterns_overrides,
            source.weight_name_prefixes,
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        index_path = Path(folder) / SAFE_WEIGHTS_INDEX_NAME
        if source.file_weight_filter is not None:
            yield from self._file_backed_weights_iterator(files, source, index_path)
            return
        yield from session.weights(
            files,
            prefixes=source.weight_name_prefixes,
            prefix=source.prefix,
            index_path=index_path,
            needs_values=self._needs_values,
            skip=lambda name: should_skip_weight(name, self.local_expert_ids),
        )
