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

logger = init_logger(__name__)


class DirectModelLoader(DefaultModelLoader):
    """Route checkpoint views through an asynchronous O_DIRECT read ring."""

    def __init__(self, load_config):
        options = dict(load_config.model_loader_extra_config)
        self.io_threads = options.pop("io_threads", 8)
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
