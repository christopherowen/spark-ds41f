# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Engram with readonly accepted history and global ceil-row TP shards."""

import os
from contextlib import suppress
from functools import lru_cache

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_transfer import copy_weight, get_file_tensor_source
from vllm.models.deepseek_v4_1.engram_rows import (
    DEFAULT_CACHE_ROWS,
    PACKED_ROW_BYTES,
    EngramDiskRows,
    MappedHostAllocation,
    column_compaction_map,
    pack_engram_columns,
    packed_table_bytes,
)
from vllm.models.deepseek_v4_1.tilelang import engram_hash as native
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource

_FP8_BLOCK_ROWS = 32


class _PaddedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear whose output need not divide by the TP size.

    The output is padded to a multiple of tp_size x 32 rows so every rank owns
    whole block-FP8 scale blocks. The last rank's missing checkpoint rows are
    zero-filled through the loader's existing ``allow_tp_padding`` path (a
    narrowed copy the B12X checkpoint loader supports), so the padded rows
    compute exactly zero, and the gathered output is sliced back to the
    logical width.
    """

    def __init__(self, input_size, output_size, **kwargs):
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        quantum = tp_size * _FP8_BLOCK_ROWS
        padded = -(-output_size // quantum) * quantum
        self.logical_output_size = output_size
        self.padded_output_size = padded
        self.sp_shards = None
        super().__init__(input_size, padded, **kwargs)
        if self.bias is not None or self.return_bias:
            raise ValueError("the padded Engram projection is bias-free")
        if padded != output_size:
            for name in ("weight", "weight_scale_inv", "weight_scale"):
                param = getattr(self, name, None)
                if param is not None:
                    param.allow_tp_padding = True

    def forward(self, input_, after_product=None):
        """The gathered product; ``after_product(rows)``, when given, runs between
        this rank's GEMM and the all-gather. Bias-free: Engram's projection has
        none."""
        output = self.local_product(input_)
        if after_product is not None:
            after_product(input_.shape[0])
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output)
        if self.padded_output_size == self.logical_output_size:
            return output
        return output[..., : self.logical_output_size].contiguous().view(*output.shape[:-1], self.logical_output_size)

    def local_product(self, input_, out=None):
        """This rank's column block of the product, not gathered; written into ``out``
        when given."""
        if out is None:
            return self.quant_method.apply(self, input_, None)
        apply_into = getattr(self.quant_method, "apply_into", None)
        if apply_into is not None:
            return apply_into(self, input_, out)
        out.copy_(self.quant_method.apply(self, input_, None))
        return out

    def gather_sp_shards(self) -> None:
        """Gather every TP rank's column block of the projection so a sequence-parallel
        prefill step can project only this rank's rows locally into ``[N, L, C]``."""
        from vllm.distributed import model_parallel_is_initialized

        if self.tp_size <= 1 or not model_parallel_is_initialized():
            return
        if (
            getattr(self, "weight_scale_inv", None) is not None
            and not hasattr(self, "tilelang_weight_sf")
            and hasattr(self.quant_method, "process_weights_after_loading")
        ):
            self.quant_method.process_weights_after_loading(self)
        w_me = getattr(self, "tilelang_weight", self.weight)
        sf_me = getattr(self, "tilelang_weight_sf", None)
        all_w = tensor_model_parallel_all_gather(w_me.unsqueeze(0), dim=0)
        if sf_me is not None:
            all_sf = tensor_model_parallel_all_gather(sf_me.unsqueeze(0), dim=0)
            self.sp_shards = tuple(
                (w_me, sf_me)
                if r == self.tp_rank
                else (all_w[r].clone(), all_sf[r].clone())
                for r in range(self.tp_size)
            )
            del all_w, all_sf
        else:
            self.sp_shards = tuple(
                w_me if r == self.tp_rank else all_w[r].clone()
                for r in range(self.tp_size)
            )
            del all_w

    def local_sp_product(
        self, local_rows: torch.Tensor, local_capacity: int
    ) -> torch.Tensor:
        """Project this rank's ``local_rows`` (``[real, K]``) through every TP rank's
        column block into ``[N, local_capacity, C]`` locally, without a collective."""
        shards = self.sp_shards
        assert shards is not None
        out = torch.empty(
            (len(shards), local_capacity, self.output_size_per_partition),
            dtype=local_rows.dtype,
            device=local_rows.device,
        )
        real = local_rows.shape[0]
        if real == 0:
            return out
        apply_shards = getattr(self.quant_method, "apply_shards", None)
        if apply_shards is not None:
            return apply_shards(self, local_rows, shards, out)
        for s, shard in enumerate(shards):
            w_s = shard[0] if isinstance(shard, tuple) else shard
            torch.matmul(local_rows, w_s.t(), out=out[s, :real])
        return out


def project_rows(wkv, rows, after_projection=None):
    """Engram's projection of ``rows``. ``after_projection(rows)``, when given, runs
    between the GEMM and a column-parallel projection's all-gather (the model's NEXT
    L2 prefetch window, ``build_l2_prefetch``), else after the GEMM."""
    if isinstance(wkv, _PaddedColumnParallelLinear):
        return wkv(rows, after_product=after_projection)
    kv = wkv(rows)
    if after_projection is not None:
        after_projection(rows.shape[0])
    return kv


logger = init_logger(__name__)
DEAD_ID = -1

# How long a forward's gather waits for a step's disk rows before it stops the
# GPU with a record of the step (a liveness bound; the rows normally arrive
# within a millisecond).
ENGRAM_ROWS_TIMEOUT_S = (
    float(os.environ.get("SPARK_DS41F_ENGRAM_ROWS_TIMEOUT_MS", "10000")) / 1000
)
# Tests only: delay every disk read, to show the forward waits for its rows.
ENGRAM_READ_DELAY_S = (
    float(os.environ.get("SPARK_DS41F_ENGRAM_READ_DELAY_MS", "0")) / 1000
)
# Debug only (off and free in serving): every row ID the host hashed early must equal
# the GPU's, or the step fails with both.
ENGRAM_CHECK_EARLY_IDS = (
    os.environ.get("SPARK_DS41F_ENGRAM_CHECK_EARLY_IDS", "0") == "1"
)
# Debug only (off and free in serving: the wait kernel's timing is compiled in only
# when set): every disk table's reader records each step's timeline, and the GPU each
# wait's length; a thread appends them every few seconds to JSON lines in
# SPARK_DS41F_ENGRAM_TIMELINE_DIR (<host>-tp<rank>-layer<id>.jsonl).
ENGRAM_TIMELINE = os.environ.get("SPARK_DS41F_ENGRAM_TIMELINE", "0") == "1"
ENGRAM_TIMELINE_DIR = os.environ.get(
    "SPARK_DS41F_ENGRAM_TIMELINE_DIR", "/cache/spark-ds41f/engram-timeline"
)
ENGRAM_PACKED_DIR = os.environ.get("SPARK_DS41F_ENGRAM_PACKED_DIR", "")
_ENGRAM_ROW_CACHE_ENV = os.environ.get("SPARK_DS41F_ENGRAM_ROW_CACHE_ROWS")
ENGRAM_ROW_CACHE_ROWS: int | None = (
    int(_ENGRAM_ROW_CACHE_ENV) if _ENGRAM_ROW_CACHE_ENV is not None else None
)
_timeline_tables: list = []


def _dump_timelines() -> None:
    import json
    import socket
    import time

    os.makedirs(ENGRAM_TIMELINE_DIR, exist_ok=True)
    host = socket.gethostname()
    while True:
        time.sleep(5)
        for rows, rank, layer in list(_timeline_tables):
            records = rows.timeline()
            if records:
                name = f"{host}-tp{rank}-layer{layer}.jsonl"
                path = os.path.join(ENGRAM_TIMELINE_DIR, name)
                with open(path, "a") as out:
                    out.writelines(json.dumps(record) + "\n" for record in records)


def _record_timeline(rows, rank: int, layer: int) -> None:
    if not _timeline_tables:
        import threading

        threading.Thread(
            target=_dump_timelines, name="engram-timeline", daemon=True
        ).start()
    _timeline_tables.append((rows, rank, layer))


@lru_cache(maxsize=2)
def _token_map(path, revision, trust_remote_code):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, revision=revision, trust_remote_code=trust_remote_code
    )
    return native.build_compressed_token_map(tokenizer)


class EngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.geometry = native.build_geometry(
            layer_ids=self.layer_ids,
            base_table_size=config.engram_vocab_size,
            compressed_vocab_size=config.engram_compressed_vocab_size,
        )
        if (self.max_ngram_size, self.n_heads, self.head_dim) != (4, 8, 256):
            raise ValueError(
                "V4.1 Engram requires four-gram/eight-head/dim256 geometry"
            )
        if self.geometry.num_embeddings != self.num_embeddings:
            raise ValueError(
                "Engram checkpoint table rows do not match global hash geometry"
            )
        vc = get_current_vllm_config()
        mc = vc.model_config
        engram_config = vc.engram_config
        self.table_memory = (
            engram_config.table_memory if engram_config is not None else "device"
        )
        self.projection_tp = (
            engram_config.projection_tp if engram_config is not None else False
        )
        if (
            self.table_memory == "disk"
            and vc.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Disk Engram requires pipeline_parallel_size=1")
        token_map, compressed_size = _token_map(
            mc.tokenizer, mc.revision, mc.trust_remote_code
        )
        if compressed_size != config.engram_compressed_vocab_size:
            raise ValueError("Engram tokenizer compressed vocabulary mismatch")
        self.token_map = token_map
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.caps = tuple(
            native.Caps(
                device=device,
                max_tokens=vc.scheduler_config.max_num_batched_tokens,
                max_seqs=vc.scheduler_config.max_num_seqs,
                max_requests=vc.scheduler_config.max_num_seqs,
                vocab_size=config.vocab_size,
                layer_id=layer,
                tp_size=get_tensor_model_parallel_world_size(),
                tp_rank=get_tensor_model_parallel_rank(),
            )
            for layer in self.layer_ids
        )
        # Both disk and resident tables gather/lookup with TileLang kernels.
        self.lookup_plans = (None,) * len(self.caps)

    @classmethod
    def from_config(cls, config):
        return cls(config) if getattr(config, "engram_layer_ids", None) else None


class NgramHashState(nn.Module):
    def __init__(self, vllm_config, layout, swa_cache_module):
        super().__init__()
        from vllm.models.deepseek_v4_1.tilelang.engram_hash import geometry_tensors

        self.layout = layout
        self.lookback_depth = 3
        self.use_slot_cache = False
        c = layout.caps[0]
        for name in ("num_seqs", "num_tokens"):
            self.register_buffer(
                name,
                torch.empty((1,), dtype=torch.int32, device=c.device),
                persistent=False,
            )
        # The TileLang hash reads the tokenizer map and geometry directly.
        self._tl_token_map = torch.tensor(
            layout.token_map, dtype=torch.int64, device=c.device
        )
        self._tl_geometry = geometry_tensors(layout.geometry, c.device)

    def ensure_cache(self):
        return True  # Hash history belongs to the runner, never the KV slot pool.

    def declare_kernels(self) -> None:
        """Declare the TileLang hash of every Engram layer. The runner's token ids
        and the model state's lookback window are int32 (profiling's window takes
        the ids' dtype)."""
        from vllm.models.deepseek_v4_1.tilelang import prepare
        from vllm.models.deepseek_v4_1.tilelang.engram_hash import engram_hash_kernel

        call = prepare.call(
            engram_hash_kernel,
            self._tl_geometry[0].shape[0],
            self._tl_token_map.numel(),
            "int32",
            "int32",
        )
        prepare.declare(lambda: [call])

    def run_native(self, ids, starts, history, out):
        """Hash ``ids`` for every Engram layer into ``out`` [T, layers, 24]; the
        hash finds the image tokens (the sentinel's ids) itself."""
        from vllm.models.deepseek_v4_1.tilelang.engram_hash import engram_hash

        retain_cuda_graph_capture_resource(self)
        c = self.layout.caps[0]
        seqs = starts.numel() - 1
        if ids.numel() > c.max_tokens or seqs > c.max_seqs:
            raise ValueError("Engram live rows exceed preallocated capacity")
        if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] < seqs:
            raise ValueError(
                "Engram requires chronological accepted history [requests,3]"
            )
        engram_hash(
            ids,
            starts,
            history,
            self._tl_token_map,
            *self._tl_geometry,
            out,
            self.num_seqs,
            self.num_tokens,
        )

    def forward(
        self,
        input_ids,
        positions,
        query_start_loc,
        lookback_token_ids,
        lookback_dead_mask=None,
        slot_mapping=None,
        block_table=None,
    ):
        out = torch.empty(
            (input_ids.numel(), len(self.layout.layer_ids), 24),
            dtype=torch.int64,
            device=input_ids.device,
        )
        self.run_native(input_ids, query_start_loc, lookback_token_ids, out)
        return out


def _read_table_rows(destination, source, start):
    """Read only this shard, straight into its final mapped CPU allocation."""
    if destination.device.type != "cpu" or not destination.is_contiguous():
        raise ValueError("File-backed Engram loading requires contiguous CPU storage")
    row_bytes = destination.shape[1] * destination.element_size()
    data = memoryview(destination.view(torch.uint8).numpy()).cast("B")
    # Unbuffered readinto has no payload-sized Python/NumPy allocation, and a
    # bounded syscall size also handles tables larger than Linux's read limit.
    start_byte = source.offset + start * row_bytes
    drop_cache = (
        os.environ.get("SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE", "1") != "0"
        and hasattr(os, "posix_fadvise")
        and hasattr(os, "POSIX_FADV_DONTNEED")
    )
    with open(source.path, "rb", buffering=0) as checkpoint:
        fd = checkpoint.fileno()
        checkpoint.seek(start_byte)
        offset = 0
        while offset < len(data):
            chunk_start = start_byte + offset
            count = checkpoint.readinto(data[offset : offset + (8 << 20)])
            if not count:
                raise OSError(
                    f"Short read loading Engram table from {source.path}: "
                    f"{offset} of {len(data)} local bytes"
                )
            offset += count
            if drop_cache:
                aligned = (chunk_start // 4096) * 4096
                done_aligned = ((chunk_start + count) // 4096) * 4096
                if done_aligned > aligned:
                    with suppress(OSError):
                        os.posix_fadvise(
                            fd, aligned, done_aligned - aligned, os.POSIX_FADV_DONTNEED
                        )
        if drop_cache and len(data) > 0:
            aligned_start = (start_byte // 4096) * 4096
            aligned_end = start_byte + len(data)
            with suppress(OSError):
                os.posix_fadvise(
                    fd,
                    aligned_start,
                    aligned_end - aligned_start,
                    os.POSIX_FADV_DONTNEED,
                )


def _load_table(param, loaded_weight):
    source = get_file_tensor_source(loaded_weight)
    shape = source.shape if source is not None else loaded_weight.shape
    dtype = source.dtype if source is not None else loaded_weight.dtype
    if shape != (param.global_rows, *param.shape[1:]):
        raise ValueError("Engram source must be the unpadded global checkpoint table")
    dtypes = (
        (torch.uint8, torch.float8_e8m0fnu)
        if param.dtype == torch.uint8
        else (torch.float8_e4m3fn,)
    )
    if dtype not in dtypes:
        raise TypeError(f"Invalid Engram table dtype: {dtype}")
    destination = getattr(param, "load_view", param.data)
    start = param.shard_start
    count = max(0, min(param.shape[0], param.global_rows - start))
    if count < param.shape[0]:
        destination[count:].zero_()
    if count:
        if source is not None:
            _read_table_rows(destination[:count], source, start)
        else:
            copy_weight(
                destination[:count].view(torch.uint8),
                loaded_weight[start : start + count].view(torch.uint8),
            )


def column_shard(geometry, layer_index: int, tp_size: int, tp_rank: int):
    """A disk table's shard on whole hash columns, or None: the columns rank
    ``tp_rank`` reads, every ``tp_size``-th from ``tp_rank``.

    A layer's table is its 24 hash columns' prime-sized ranges back to back; heads
    0-7 hash 2-grams, 8-15 3-grams and 16-23 4-grams, and a longer n-gram repeats
    less, so a step reads more distinct rows of it (1 : 1.35 : 1.54 on source text).
    Interleaving gives every rank columns of each order (TP3: 3/3/2, 3/2/3, 2/3/3;
    TP4: 2/2/2; TP5: 2/2/1, 2/1/2, 1/2/2, 1/2/2, 2/1/1), where contiguous columns
    gave one rank every 4-gram head and the slowest reads. A rank's gathered columns
    are joined by an all-gather and put back in column order (``interleaved_rows``)."""
    columns = len(geometry.offsets[layer_index])
    if tp_size <= 0 or tp_size > columns:
        return None
    return tuple(range(tp_rank, columns, tp_size))


def interleaved_rows(local: torch.Tensor) -> torch.Tensor:
    """Rows ``[T, 24 * 256]`` in column order from every rank's columns: ``local`` is
    this rank's ``[T, ceil(24 / tp) * 256]``, columns ``rank, rank + tp, ...`` side by
    side. All-gathered as 256-wide rows along the last dimension, rank r's k-th block
    of token t lands at column k * tp + r of the token's row: the column order, with
    no copy before or after the collective when ``tp`` divides 24 (and a narrow slice
    of the trailing padded columns when ``24 % tp != 0``)."""
    tokens = local.shape[0]
    gathered = tensor_model_parallel_all_gather(local.view(-1, 256), dim=-1)
    rows = gathered.view(tokens, -1)
    if rows.shape[1] == 24 * 256:
        return rows
    return rows[:, : 24 * 256].contiguous().view(tokens, 24 * 256)


def interleaved_sp_rows(local: torch.Tensor, sp_rows) -> torch.Tensor:
    """This rank's sequence-parallel rows ``[real, 24 * 256]`` in column order from
    every rank's columns: ``local`` is this rank's ``[T, ceil(24 / tp) * 256]``,
    columns ``rank, rank + tp, ...`` side by side. Exchanged all-to-all by row block
    so each rank receives only its ``real`` local rows from every rank, then placed
    side by side as 256-wide column blocks."""
    blocks = sp_rows.exchange_rows(local)
    real = sp_rows.real_rows
    if real == 0:
        return local.new_empty((0, 24 * 256))
    if real < sp_rows.local_rows:
        blocks = blocks[:, :real].contiguous()
    world_size = blocks.shape[0]
    shard_shape = (real * (blocks.shape[2] // 256), 256)
    flat = blocks.view(world_size * shard_shape[0], 256)
    from vllm.models.deepseek_v4_1.tilelang.collectives import concat_rank_shards

    concatenated = concat_rank_shards(flat, world_size, shard_shape)
    if concatenated is None:
        concatenated = (
            flat.view(world_size, *shard_shape)
            .movedim(0, 1)
            .reshape(shard_shape[0], world_size * 256)
        )
    rows = concatenated.view(real, -1)
    if rows.shape[1] == 24 * 256:
        return rows
    return rows[:, : 24 * 256].contiguous().view(real, 24 * 256)


class ParallelEngramEmbedding(nn.Module):
    def __init__(self, plan, caps, geometry, table_memory="device"):
        super().__init__()
        self.plan = plan
        self.caps = caps
        self.geometry = geometry
        self.layer_index = geometry.layer_ids.index(caps.layer_id)
        self.table_rows = geometry.num_embeddings[self.layer_index]
        self.shard_rows = (self.table_rows + caps.tp_size - 1) // caps.tp_size
        self.shard_start = caps.tp_rank * self.shard_rows
        self.shard_end = (caps.tp_rank + 1) * self.shard_rows
        # A disk table's rank reads whole hash columns (every tp-th) when the TP size
        # divides them, so the ranks' rows are joined by an all-gather of their
        # columns, not summed (resident tables keep equal row shards).
        self.columns = None
        self.max_columns = 0
        self.column_mask = (1 << 24) - 1
        total_columns = len(
            geometry.offsets[geometry.layer_ids.index(caps.layer_id)]
        )
        columns = (
            column_shard(
                geometry,
                geometry.layer_ids.index(caps.layer_id),
                caps.tp_size,
                caps.tp_rank,
            )
            if table_memory == "disk"
            else None
        )
        if columns is not None:
            self.shard_start, self.shard_end = 0, self.table_rows
            self.columns = columns
            self.max_columns = (total_columns + caps.tp_size - 1) // caps.tp_size
            self.column_mask = sum(1 << column for column in columns)
        self.weight_shape = (self.shard_rows, 256)
        self.scale_shape = (self.shard_rows, 8)
        self.tp_size = caps.tp_size
        if table_memory not in ("device", "ram", "disk"):
            raise ValueError("Engram table_memory must be device, ram or disk")
        self.table_memory = table_memory
        self.disk_rows = None
        self._disk_sources = []
        self.mapped_host_nbytes = 0
        if table_memory == "ram":
            nbytes = (
                self.weight_shape[0] * self.weight_shape[1]
                + self.scale_shape[0] * self.scale_shape[1]
            )
            logger.info(
                "Engram layer %d TP rank %d: allocating %.2f GiB mapped-host "
                "RAM (packed E4M3 weights and E8M0 scales)",
                caps.layer_id,
                caps.tp_rank,
                nbytes / (1 << 30),
            )
            self._weight_allocation = None
            try:
                self._weight_allocation = MappedHostAllocation(
                    self.weight_shape,
                    torch.float8_e4m3fn,
                    caps.device,
                    write_combined=True,
                )
                self._scale_allocation = MappedHostAllocation(
                    self.scale_shape, torch.uint8, caps.device, write_combined=True
                )
            except Exception as exc:
                if self._weight_allocation is not None:
                    self._weight_allocation.close()
                raise RuntimeError(
                    f"Engram mapped-host RAM allocation failed for "
                    f"{nbytes / (1 << 30):.2f} GiB on {caps.device}; "
                    "no disk or device fallback is permitted"
                ) from exc
            self.mapped_host_nbytes = nbytes
            self.weight_load_view = self._weight_allocation.host
            self.weight_scale_load_view = self._scale_allocation.host
            self.weight = nn.Parameter(
                self._weight_allocation.device_view, requires_grad=False
            )
            self.weight_scale_inv = nn.Parameter(
                self._scale_allocation.device_view, requires_grad=False
            )
            set_weight_attrs(self.weight, {"load_view": self.weight_load_view})
            set_weight_attrs(
                self.weight_scale_inv, {"load_view": self.weight_scale_load_view}
            )
        elif table_memory != "disk":
            self.weight = nn.Parameter(
                torch.empty(
                    self.weight_shape,
                    dtype=torch.float8_e4m3fn,
                    device=caps.device,
                ),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(self.scale_shape, dtype=torch.uint8, device=caps.device),
                requires_grad=False,
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("weight_scale_inv", None)
        if table_memory != "disk":
            for param in (self.weight, self.weight_scale_inv):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": _load_table,
                        "global_rows": self.table_rows,
                        "shard_start": self.shard_start,
                    },
                )
        if table_memory == "disk":
            self.register_buffer("hashes", None, persistent=False)
            self.register_buffer("num_tokens", None, persistent=False)
        else:
            self.register_buffer(
                "hashes",
                torch.full(
                    (caps.max_tokens, 24),
                    DEAD_ID,
                    dtype=torch.int64,
                    device=caps.device,
                ),
                persistent=False,
            )
            self.register_buffer(
                "num_tokens",
                torch.empty((1,), dtype=torch.int32, device=caps.device),
                persistent=False,
            )

    def __getattr__(self, name: str):
        if name in ("hashes", "num_tokens"):
            value = self._buffers.get(name)
            if value is None:
                self._buffers["hashes"] = torch.full(
                    (self.caps.max_tokens, 24),
                    DEAD_ID,
                    dtype=torch.int64,
                    device=self.caps.device,
                )
                self._buffers["num_tokens"] = torch.empty(
                    (1,), dtype=torch.int32, device=self.caps.device
                )
                return self._buffers[name]
        return super().__getattr__(name)

    def close(self):
        """Release file-backed staging only after graph consumers are gone."""
        if self.disk_rows is not None:
            self.disk_rows.close()
            self.disk_rows = None
        for allocation in (
            getattr(self, "_scale_allocation", None),
            getattr(self, "_weight_allocation", None),
        ):
            if allocation is not None:
                allocation.close()

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            if name not in ("weight", "weight_scale_inv"):
                raise ValueError(f"Unknown Engram table weight: {name}")
            if self.table_memory != "disk":
                _load_table(getattr(self, name), value)
            else:
                source = get_file_tensor_source(value)
                if source is None:
                    raise ValueError(
                        "Disk Engram requires a file-backed checkpoint table"
                    )
                scale = name == "weight_scale_inv"
                width = self.scale_shape[1] if scale else self.weight_shape[1]
                if source.shape != (self.table_rows, width):
                    raise ValueError("Engram source must be the unpadded global table")
                dtypes = (
                    (torch.uint8, torch.float8_e8m0fnu)
                    if scale
                    else (torch.float8_e4m3fn,)
                )
                if source.dtype not in dtypes:
                    raise TypeError(f"Invalid Engram {name} dtype: {source.dtype}")
                self._disk_sources.append((source.path, source.offset, scale))
            loaded.add(name)
        return loaded

    def _disk_shard(self) -> dict[str, int]:
        """The disk reader's rows: this rank's shard of the table, its hash columns
        and the lookups of the largest step."""
        return dict(
            shard_start=min(self.shard_start, self.table_rows),
            shard_end=min(self.shard_end, self.table_rows),
            columns=self.column_mask,
            max_lookups=self.caps.max_tokens * 24,
        )

    def declare_kernels(self) -> None:
        """Declare a disk table's TileLang row kernels (the wait, the gather of its
        reader's buffers and shard, and the release) or a resident table's TileLang
        lookup kernel."""
        from vllm.models.deepseek_v4_1.tilelang import prepare
        from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
            resident_lookup_calls,
            row_kernel_calls,
        )

        if self.table_memory == "disk":
            calls = row_kernel_calls(**self._disk_shard(), out_cols=self.max_columns)
        else:
            calls = resident_lookup_calls(
                self.weight.shape[0],
                min(self.shard_start, self.table_rows),
                min(self.shard_end, self.table_rows),
                self.column_mask,
                out_cols=self.max_columns,
            )
        prepare.declare(lambda: calls)

    def open_disk_rows(self) -> None:
        """Start reading this rank's shard of the table from the checkpoint, once."""
        if self.table_memory != "disk" or self.disk_rows is not None:
            return
        planes = sorted(scale for _, _, scale in self._disk_sources)
        if planes != [False, True]:
            raise ValueError(
                "Disk Engram needs the table's weight and scale planes, once each"
            )
        cache_kw = (
            {"cache_rows": ENGRAM_ROW_CACHE_ROWS}
            if ENGRAM_ROW_CACHE_ROWS is not None
            else {}
        )
        rows = EngramDiskRows(
            device=self.caps.device,
            table_rows=self.table_rows,
            **self._disk_shard(),
            rows_timeout_s=ENGRAM_ROWS_TIMEOUT_S,
            delay_s=ENGRAM_READ_DELAY_S,
            check_early=ENGRAM_CHECK_EARLY_IDS,
            timeline=ENGRAM_TIMELINE,
            **cache_kw,
        )
        weight_path, weight_offset = next(
            (p, off) for p, off, s in self._disk_sources if not s
        )
        scale_path, scale_offset = next(
            (p, off) for p, off, s in self._disk_sources if s
        )
        packed_name = (
            f"engram-layer{self.caps.layer_id}-tp{self.caps.tp_size}"
            f"-rank{self.caps.tp_rank}.packed"
        )
        owned_columns = (
            self.columns
            if self.columns is not None
            else tuple(c for c in range(24) if (self.column_mask >> c) & 1)
        )
        col_start, col_end, col_delta, owned_starts, owned_rows = (
            column_compaction_map(
                self.geometry.offsets[self.layer_index],
                self.table_rows,
                owned_columns,
            )
        )
        expected_bytes = packed_table_bytes(int(owned_rows.sum()))
        prepacked_path = os.path.join(os.path.dirname(weight_path), packed_name)
        if ENGRAM_PACKED_DIR:
            import time

            os.makedirs(ENGRAM_PACKED_DIR, exist_ok=True)
            packed_path = os.path.join(ENGRAM_PACKED_DIR, packed_name)
            reused = (
                os.path.exists(packed_path)
                and os.path.getsize(packed_path) == expected_bytes
            )
            pack_seconds = 0.0
            if not reused:
                pack_started = time.monotonic()
                tmp_path = f"{packed_path}.tmp.{os.getpid()}"
                pack_engram_columns(
                    weight_path,
                    weight_offset,
                    scale_path,
                    scale_offset,
                    tmp_path,
                    owned_starts,
                    owned_rows,
                )
                os.replace(tmp_path, packed_path)
                pack_seconds = time.monotonic() - pack_started
            logger.info(
                "Engram packed rows layer=%d tp=%d rank=%d layout=264-v1 "
                "bytes=%d written_bytes=%d reused=%s pack_s=%.6f path=%s",
                self.caps.layer_id,
                self.caps.tp_size,
                self.caps.tp_rank,
                expected_bytes,
                0 if reused else expected_bytes,
                reused,
                pack_seconds,
                packed_path,
            )
            rows.add_packed_source(
                packed_path,
                0,
                col_start=col_start,
                col_end=col_end,
                col_delta=col_delta,
            )
        elif (
            os.path.exists(prepacked_path)
            and os.path.getsize(prepacked_path) == expected_bytes
        ):
            logger.info(
                "Engram packed rows layer=%d tp=%d rank=%d layout=264-v1 "
                "bytes=%d written_bytes=%d reused=%s pack_s=%.6f path=%s",
                self.caps.layer_id,
                self.caps.tp_size,
                self.caps.tp_rank,
                expected_bytes,
                0,
                True,
                0.0,
                prepacked_path,
            )
            rows.add_packed_source(
                prepacked_path,
                0,
                col_start=col_start,
                col_end=col_end,
                col_delta=col_delta,
            )
        else:
            for path, offset, scale in self._disk_sources:
                rows.add_source(path, offset, scale=scale)
        self.disk_rows = rows
        if ENGRAM_TIMELINE:
            _record_timeline(rows, self.caps.tp_rank, self.caps.layer_id)

    def share_hashes(self, hashes, num_tokens):
        """Read each step's row IDs (``hashes`` [max_tokens, 24], contiguous) and live
        token count from the model's buffers, which its hash writes in place, instead
        of this table's own."""
        if hashes.shape != (self.caps.max_tokens, 24) or not hashes.is_contiguous():
            raise ValueError("Engram row IDs must be a contiguous [max_tokens, 24]")
        self.hashes = hashes
        self.num_tokens = num_tokens

    def stage_rows(self, tokens, stream, early=None):
        """Queue a step's rows: hand the reader the row IDs of its ``tokens`` tokens
        (``hashes``, with the live token count in ``num_tokens``, both written before
        on ``stream``) and the IDs the host hashed (``early`` [tokens, 24] int64 on the
        host, or None). Returns at once; the forward's ``gather_rows`` waits for the
        rows."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Disk Engram staging must run outside compile/capture")
        self.disk_rows.stage(stream, self.hashes, tokens * 24, early)

    def stage_no_rows(self):
        """A dummy step: the gather writes zeros and waits for nothing."""
        self.num_tokens.zero_()

    def gather_rows(self, out):
        """Wait for the staged step's rows and dequantize them into ``out``."""
        self.disk_rows.gather(out, self.hashes, self.num_tokens)

    def lookup_native(self, indices, out, *, clear_tail: bool = True):
        from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
            engram_resident_lookup,
        )

        if indices.data_ptr() != self.hashes.data_ptr():
            self.hashes[: indices.shape[0]].copy_(indices)
            self.num_tokens.fill_(indices.shape[0])
        retain_cuda_graph_capture_resource(self)
        target = out if clear_tail else out[: indices.shape[0]]
        engram_resident_lookup(
            self.hashes,
            self.num_tokens,
            self.weight,
            self.weight_scale_inv,
            target,
            min(self.shard_start, self.table_rows),
            min(self.shard_end, self.table_rows),
            self.column_mask,
        )

    def lookup(self, indices, out, *, clear_tail: bool = True):
        self.lookup_native(indices, out, clear_tail=clear_tail)


class Engram(nn.Module):
    supports_fused_post = True

    def __init__(
        self,
        config,
        quant_config,
        layout,
        layer_hash_index,
        use_sequence_parallel=False,
        prefix="",
    ):
        super().__init__()
        if use_sequence_parallel:
            raise ValueError("V4.1 Engram consumes replicated token rows")
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        plan = layout.lookup_plans[layer_hash_index]
        caps = layout.caps[layer_hash_index]
        self.embed_tokens = ParallelEngramEmbedding(
            plan, caps, layout.geometry, layout.table_memory
        )
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        # The smallest forward the model runs sequence-parallel (0: never); the
        # model sets it.
        self.prefill_sp_min_rows = 0
        # Called with the row count between the projection's GEMM and its
        # all-gather: the model issues the layer's NEXT L2 prefetch window there
        # (``build_l2_prefetch``).
        self.l2_prefetch_after_projection = None
        projection_tp = getattr(layout, "projection_tp", False)
        projection_cls = (
            _PaddedColumnParallelLinear if projection_tp else ReplicatedLinear
        )
        self.wkv = projection_cls(
            6144,
            self.dim * (self.hc_mult + 1),
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            **({"gather_output": True} if projection_tp else {}),
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty_like(self.q_weight), requires_grad=False
        )
        self.register_buffer(
            "norm_weights",
            torch.empty(self.hc_mult * self.dim, dtype=torch.float32),
            persistent=False,
        )
        # Zeroed: a gather writes only its own step's rows. A disk table sharded on
        # hash columns gathers only its own columns, side by side. Disk-backed layers
        # gather and immediately project within their own forward, so they share one
        # staged_rows buffer across layers on the layout.
        columns = self.embed_tokens.columns
        max_columns = self.embed_tokens.max_columns or (
            len(columns) if columns is not None else 0
        )
        staged_rows_shape = (
            caps.max_tokens,
            6144 if columns is None else max_columns * 256,
        )
        shared_staged_rows = (
            getattr(layout, "_shared_disk_staged_rows", None)
            if layout.table_memory == "disk"
            else None
        )
        if (
            shared_staged_rows is None
            or shared_staged_rows.shape != staged_rows_shape
            or shared_staged_rows.device != torch.device(caps.device)
        ):
            shared_staged_rows = torch.zeros(
                *staged_rows_shape,
                dtype=torch.bfloat16,
                device=caps.device,
            )
            if layout.table_memory == "disk":
                layout._shared_disk_staged_rows = shared_staged_rows
        self.register_buffer(
            "staged_rows",
            shared_staged_rows,
            persistent=False,
        )

    def get_b12x_preparation_units(self, layer=None, workload=None):
        return ()

    def declare_kernels(self) -> None:
        """Declare the TileLang gate (both standalone and fused with ``mhc.post``):
        on packed rows always, and on the column blocks of a column-parallel
        projection when the model runs prefill sequence parallel (the blocked gate,
        one block per rank). The model passes the step's int32 token ids, the gate
        finding the image sentinel itself."""
        from vllm.models.deepseek_v4_1.tilelang import prepare
        from vllm.models.deepseek_v4_1.tilelang.engram import (
            engram_gate_blocked_kernel,
            engram_gate_kernel,
            engram_post_gate_blocked_kernel,
            engram_post_gate_kernel,
        )

        calls = [
            prepare.call(engram_gate_kernel, self.hc_mult, self.dim, self.eps, "int32"),
            prepare.call(
                engram_post_gate_kernel, self.hc_mult, self.dim, self.eps, "int32"
            ),
        ]
        sequence_parallel = (
            0 < self.prefill_sp_min_rows <= self.embed_tokens.caps.max_tokens
        )
        world_size = get_tensor_model_parallel_world_size()
        if sequence_parallel:
            # SP's trailing pad rows pass through (forward).
            from vllm.models.deepseek_v4_1.tilelang import columns

            key = columns.copy_columns_key(self.hc_mult * self.dim, torch.bfloat16)
            calls.append(prepare.call(columns.copy_columns_kernel, *key))
            if world_size > 1:
                calls.append(
                    prepare.call(
                        engram_post_gate_kernel,
                        self.hc_mult,
                        self.dim,
                        self.eps,
                        "int32",
                        world=world_size,
                    )
                )
        if isinstance(self.wkv, _PaddedColumnParallelLinear) and (
            sequence_parallel
            or (
                getattr(self.wkv, "gather_output", False)
                and getattr(self.wkv, "tp_size", 1) > 1
            )
        ):
            block_cols = self.wkv.output_size_per_partition
            calls.append(
                prepare.call(
                    engram_gate_blocked_kernel,
                    self.hc_mult,
                    self.dim,
                    self.eps,
                    "int32",
                    world_size,
                    block_cols,
                )
            )
            calls.append(
                prepare.call(
                    engram_post_gate_blocked_kernel,
                    self.hc_mult,
                    self.dim,
                    self.eps,
                    "int32",
                    world_size,
                    block_cols,
                )
            )
            if sequence_parallel and world_size > 1:
                calls.append(
                    prepare.call(
                        engram_post_gate_blocked_kernel,
                        self.hc_mult,
                        self.dim,
                        self.eps,
                        "int32",
                        world_size,
                        block_cols,
                        world=world_size,
                    )
                )
        prepare.declare(lambda: calls)

    def process_weights_after_loading(self):
        self.norm_weights.copy_(
            (self.q_weight.float() * self.k_weight.float()).flatten()
        )
        self.embed_tokens.open_disk_rows()
        sequence_parallel = (
            0 < self.prefill_sp_min_rows <= self.embed_tokens.caps.max_tokens
        )
        if sequence_parallel and isinstance(self.wkv, _PaddedColumnParallelLinear):
            self.wkv.gather_sp_shards()

    def stage_rows(self, tokens, stream, early=None):
        """Queue the disk rows of this step's ``tokens`` tokens (their row IDs in
        ``embed_tokens.hashes``; ``early``: the IDs the host hashed); the forward
        gathers them."""
        self.invalidate_disk_output()
        self.embed_tokens.stage_rows(tokens, stream, early)
        self._disk_prepared_tokens = tokens
        self._disk_prepared = True

    def prepare_embeddings(self, hash_ids):
        self.embed_tokens.lookup(hash_ids, self.staged_rows, clear_tail=False)

    def invalidate_disk_output(self, *, clear=False):
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        if clear:
            self.staged_rows.zero_()

    def prepare_dummy_output(self, num_tokens):
        self.embed_tokens.stage_no_rows()
        self._disk_prepared_tokens = num_tokens
        self._disk_prepared = True

    def forward(
        self,
        hidden_states,
        hash_ids,
        token_mask=None,
        sp_rows=None,
        *,
        previous_output=None,
        post_mix=None,
        res_mix=None,
    ):
        """Gate ``hidden_states`` with this layer's Engram rows.

        ``token_mask`` is the step's token ids (the image sentinel's rows pass
        through). When ``previous_output``, ``post_mix`` and ``res_mix`` are
        given, reconstructs ``hidden_states`` via ``mhc.post`` inside the gate
        kernel in a single launch without writing or reading an intermediate
        multi-stream residual tensor.

        With ``sp_rows`` (prefill sequence parallelism) ``hidden_states`` holds
        this rank's rows while ``hash_ids`` and ``token_mask`` cover every
        token; the lookup runs on every row, and with a column-parallel
        projection each rank receives only its own rows (exchanging disk hash
        columns before projecting locally through ``wkv``'s column blocks, or
        exchanging ``wkv``'s column blocks all-to-all).
        """
        num_tokens = hash_ids.shape[0]
        if self.embed_tokens.table_memory == "disk" and (
            not self._disk_prepared or self._disk_prepared_tokens < num_tokens
        ):
            raise RuntimeError("Disk Engram output is not prepared")
        if self.embed_tokens.table_memory == "disk":
            self.embed_tokens.gather_rows(self.staged_rows[:num_tokens])
        # With a column-parallel projection, kv is every rank's column block of
        # the rows, [N, L, C]: the gate reads the rank-major blocks in place.
        tp_blocked_wkv = (
            sp_rows is None
            and isinstance(self.wkv, _PaddedColumnParallelLinear)
            and getattr(self.wkv, "gather_output", False)
            and getattr(self.wkv, "tp_size", 1) > 1
        )
        blocked = (
            sp_rows is not None and isinstance(self.wkv, _PaddedColumnParallelLinear)
        ) or tp_blocked_wkv
        sp_local_wkv = (
            sp_rows is not None
            and blocked
            and self.embed_tokens.columns is not None
            and getattr(self.wkv, "sp_shards", None) is not None
        )
        if sp_local_wkv:
            # Each rank exchanges its hash columns by row block and interleaves
            # only its own sequence-parallel rows before projecting locally.
            rows = interleaved_sp_rows(self.staged_rows[:num_tokens], sp_rows)
        elif self.embed_tokens.columns is not None:
            # This rank's gathered rows are its hash columns, every tp-th.
            rows = interleaved_rows(self.staged_rows[:num_tokens])
        else:
            rows = tensor_model_parallel_all_reduce(
                self.staged_rows[:num_tokens]
            )
        state = (
            hidden_states if previous_output is not None else hidden_states.flatten(1)
        )
        out = None

        def prepare_local():
            nonlocal previous_output, out
            if previous_output is not None:
                from vllm.models.deepseek_v4_1.tilelang.collectives import (
                    RankOrderParts,
                )

                if isinstance(previous_output, RankOrderParts):
                    previous_output = previous_output.packed
            out = torch.empty_like(state)
            if sp_rows is not None:
                real = sp_rows.real_rows
                if real < state.shape[0]:
                    if previous_output is not None:
                        from vllm.models.deepseek_v4_1.tilelang.mhc import _mhc_post

                        _mhc_post(
                            previous_output[:, real:]
                            if previous_output.ndim == 3
                            else previous_output[real:],
                            hidden_states[real:],
                            post_mix[real:],
                            res_mix[real:],
                            out[real:],
                        )
                    else:
                        from vllm.models.deepseek_v4_1.tilelang import columns

                        columns.copy_columns(state[real:], out[real:])

        if sp_local_wkv:
            kv = self.wkv.local_sp_product(rows, sp_rows.local_rows)
            del rows
            if self.l2_prefetch_after_projection is not None:
                self.l2_prefetch_after_projection(num_tokens)
            prepare_local()
            kv_start = 0
        elif tp_blocked_wkv:
            block_cols = (
                getattr(self.wkv, "output_size_per_partition", None)
                or self.wkv.weight.shape[0]
            )
            local_kv = self.wkv.local_product(rows)
            del rows
            if self.l2_prefetch_after_projection is not None:
                self.l2_prefetch_after_projection(num_tokens)
            prepare_local()
            kv = tensor_model_parallel_all_gather(local_kv, dim=0).view(
                self.wkv.tp_size, num_tokens, block_cols
            )
            del local_kv
            kv_start = 0
        elif blocked:
            from vllm.models.deepseek_v4_1 import sp_prefill

            block_cols = (
                getattr(self.wkv, "output_size_per_partition", None)
                or self.wkv.weight.shape[0]
            )
            after_proj = self.l2_prefetch_after_projection

            def after_blocked():
                if after_proj is not None:
                    after_proj(num_tokens)
                prepare_local()

            kv = sp_rows.project_exchange_rows(
                lambda start, stop, dest: self.wkv.local_product(
                    rows[start:stop], out=dest
                ),
                (block_cols,),
                rows.dtype,
                rows.device,
                slices=sp_prefill.engram_slices(num_tokens),
                then=after_blocked,
            )
            del rows
            kv_start = 0
        else:
            if sp_rows is not None and not isinstance(self.wkv, ColumnParallelLinear):
                # Replicated projection under SP: each rank projects only its real rows.
                rows = rows[sp_rows.start : sp_rows.start + sp_rows.real_rows]
                kv_start = 0
            else:
                kv_start = 0 if sp_rows is None else sp_rows.start
            if rows.shape[0] > 0 or sp_rows is None:
                kv = project_rows(self.wkv, rows, self.l2_prefetch_after_projection)
            else:
                if self.l2_prefetch_after_projection is not None:
                    self.l2_prefetch_after_projection(0)
                kv = None
            del rows
            prepare_local()
        if previous_output is not None:
            from vllm.models.deepseek_v4_1.tilelang.engram import (
                engram_post_gate,
                engram_post_gate_blocked,
            )

            residual_in = hidden_states
            mix_out = out
            if sp_rows is not None:
                start, real = sp_rows.start, sp_rows.real_rows
                if real == 0:
                    return out
                previous_output = (
                    previous_output[:, :real]
                    if previous_output.ndim == 3
                    else previous_output[:real]
                )
                residual_in = hidden_states[:real]
                post_mix = post_mix[:real]
                res_mix = res_mix[:real]
                mix_out = out[:real]
                if not blocked:
                    kv = kv[kv_start : kv_start + real]
                if token_mask is not None:
                    token_mask = token_mask[start : start + real]
            post_gate = engram_post_gate_blocked if blocked else engram_post_gate
            post_gate(
                previous_output,
                residual_in,
                post_mix,
                res_mix,
                kv,
                self.norm_weights,
                token_mask,
                mix_out,
                self.eps,
                self.hc_mult,
            )
            return out
        mix_out = out
        if sp_rows is not None:
            # Gate the real local rows; the trailing pad rows pass through.
            start, real = sp_rows.start, sp_rows.real_rows
            if real == 0:
                return out.view_as(hidden_states)
            state, mix_out = state[:real], out[:real]
            if not blocked:  # the blocked gate reads only the first `real` rows
                kv = kv[kv_start : kv_start + real]
            if token_mask is not None:
                token_mask = token_mask[start : start + real]
        from vllm.models.deepseek_v4_1.tilelang.engram import (
            engram_gate,
            engram_gate_blocked,
        )

        # The model passes the token ids: the sentinel's rows pass.
        gate = engram_gate_blocked if blocked else engram_gate
        gate(state, kv, self.norm_weights, token_mask, mix_out, self.eps, self.hc_mult)
        return out.view_as(hidden_states)


# Debug row trace taps (boot-differences overlay): None unless SPARK_DS41F_ROW_TRACE_DIR.
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    ROW_TRACE as _ROW_TRACE,
)

if _ROW_TRACE is not None:
    _untraced_engram_forward = Engram.forward

    def _traced_engram_forward(self, hidden_states, *args, **kwargs):
        out = _untraced_engram_forward(self, hidden_states, *args, **kwargs)
        n = int(out.shape[0]) if out.dim() else 0
        staged = getattr(self, "staged_rows", None)
        rows = staged[:n] if isinstance(staged, torch.Tensor) and staged.shape[0] >= n else out
        _ROW_TRACE.record(8, rows, out)
        return out

    Engram.forward = _traced_engram_forward
