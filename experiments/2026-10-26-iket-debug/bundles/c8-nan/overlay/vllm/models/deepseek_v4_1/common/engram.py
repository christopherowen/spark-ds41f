# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Engram with readonly accepted history and global ceil-row TP shards."""

import os
from functools import lru_cache

import torch
import torch.nn as nn
from b12x.preparation import PreparedCall
from b12x.sequence import engram as native

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
from vllm.models.deepseek_v4_1.engram_rows import EngramDiskRows, MappedHostAllocation
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    set_b12x_preparation_provider,
)
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
        super().__init__(input_size, padded, **kwargs)
        if padded != output_size:
            for name in ("weight", "weight_scale_inv", "weight_scale"):
                param = getattr(self, name, None)
                if param is not None:
                    param.allow_tp_padding = True

    def forward(self, input_):
        output = super().forward(input_)
        if self.padded_output_size == self.logical_output_size:
            return output
        return output[..., : self.logical_output_size].contiguous()

    def local_product(self, input_):
        """This rank's column block of the product, not gathered."""
        return self.quant_method.apply(self, input_, None)


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
        # Disk tables gather with the TileLang kernel; only resident tables use
        # the B12X lookup.
        self.lookup_plans = tuple(
            None
            if self.table_memory == "disk"
            else native.plan(
                caps,
                token_map=token_map,
                geometry=self.geometry,
                invocation={
                    "operation": "lookup",
                    "compact_rows": False,
                    "resident_scales": False,
                },
            )
            for caps in self.caps
        )

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
    with open(source.path, "rb", buffering=0) as checkpoint:
        checkpoint.seek(source.offset + start * row_bytes)
        offset = 0
        while offset < len(data):
            count = checkpoint.readinto(data[offset : offset + (8 << 20)])
            if not count:
                raise OSError(
                    f"Short read loading Engram table from {source.path}: "
                    f"{offset} of {len(data)} local bytes"
                )
            offset += count


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
    TP4: 2/2/2), where contiguous columns gave one rank every 4-gram head and the
    slowest reads. A rank's gathered columns are joined by an all-gather and put back
    in column order (``interleaved_rows``). Requires a TP size that divides 24."""
    columns = len(geometry.offsets[layer_index])
    if columns % tp_size:
        return None
    return tuple(range(tp_rank, columns, tp_size))


def interleaved_rows(local: torch.Tensor) -> torch.Tensor:
    """Rows ``[T, 24 * 256]`` in column order from every rank's columns: ``local`` is
    this rank's ``[T, (24 / tp) * 256]``, columns ``rank, rank + tp, ...`` side by
    side. All-gathered as 256-wide rows along the last dimension, rank r's k-th block
    of token t lands at column k * tp + r of the token's row: the column order, with
    no copy before or after the collective (the one-shot all-gather writes the layout
    directly). Exact copies: the same values as one reader's rows."""
    tokens = local.shape[0]
    gathered = tensor_model_parallel_all_gather(local.view(-1, 256), dim=-1)
    return gathered.view(tokens, 24 * 256)


class ParallelEngramEmbedding(nn.Module):
    def __init__(self, plan, caps, geometry, table_memory="device"):
        super().__init__()
        self.plan = plan
        self.caps = caps
        self.table_rows = geometry.num_embeddings[
            geometry.layer_ids.index(caps.layer_id)
        ]
        self.shard_rows = (self.table_rows + caps.tp_size - 1) // caps.tp_size
        self.shard_start = caps.tp_rank * self.shard_rows
        self.shard_end = (caps.tp_rank + 1) * self.shard_rows
        # A disk table's rank reads whole hash columns (every tp-th) when the TP size
        # divides them, so the ranks' rows are joined by an all-gather of their
        # columns, not summed (resident tables keep equal row shards).
        self.columns = None
        self.column_mask = (1 << 24) - 1
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
        self.register_buffer(
            "hashes",
            torch.full(
                (caps.max_tokens, 24), DEAD_ID, dtype=torch.int64, device=caps.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "num_tokens",
            torch.empty((1,), dtype=torch.int32, device=caps.device),
            persistent=False,
        )

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
        reader's buffers and shard, and the release)."""
        if self.table_memory != "disk":
            return
        from vllm.models.deepseek_v4_1.tilelang import prepare
        from vllm.models.deepseek_v4_1.tilelang.engram_rows import row_kernel_calls

        calls = row_kernel_calls(**self._disk_shard())
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
        rows = EngramDiskRows(
            device=self.caps.device,
            table_rows=self.table_rows,
            **self._disk_shard(),
            rows_timeout_s=ENGRAM_ROWS_TIMEOUT_S,
            delay_s=ENGRAM_READ_DELAY_S,
            check_early=ENGRAM_CHECK_EARLY_IDS,
            timeline=ENGRAM_TIMELINE,
        )
        for path, offset, scale in self._disk_sources:
            rows.add_source(path, offset, scale=scale)
        self.disk_rows = rows
        if ENGRAM_TIMELINE:
            _record_timeline(rows, self.caps.tp_rank, self.caps.layer_id)

    def share_hashes(self, hashes, num_tokens):
        """Read each step's row IDs (``hashes`` [max_tokens, 24], contiguous) and live
        token count from the model's buffers, which its hash writes in place, instead
        of this table's own."""
        if hashes.shape != self.hashes.shape or not hashes.is_contiguous():
            raise ValueError("Engram row IDs must be a contiguous [max_tokens, 24]")
        self.hashes = hashes
        self.num_tokens = num_tokens

    def stage_rows(self, tokens, stream, early=None):
        """Queue a step's rows: hand the reader the row IDs of its ``tokens`` tokens
        (``hashes``, with the live token count in ``num_tokens``, both written before
        on ``stream``) and the IDs the host hashed (``early`` [tokens, 24] int64 on the
        host, or None). Returns at once; the forward's ``gather_rows`` waits for the
        rows."""
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Disk Engram staging must run outside compile/capture")
        self.disk_rows.stage(stream, self.hashes, tokens * 24, early)

    def stage_no_rows(self):
        """A dummy step: the gather writes zeros and waits for nothing."""
        self.num_tokens.zero_()

    def gather_rows(self, out):
        """Wait for the staged step's rows and dequantize them into ``out``."""
        self.disk_rows.gather(out, self.hashes, self.num_tokens)

    def lookup_native(self, indices, out):
        if self.plan.prepared is None:
            raise PreparationResourceUnavailableError(
                "Engram lookup plan must be prepared before forwarding"
            )
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.fill_(indices.shape[0])
        binding = native.bind_lookup(
            self.plan,
            weight=self.weight,
            scales=self.weight_scale_inv,
            hash_ids=self.hashes,
            num_tokens=self.num_tokens,
            out=out,
        )
        retain_cuda_graph_capture_resource(binding)
        if self.mapped_host_nbytes:
            retain_cuda_graph_capture_resource(self)
        native.run_lookup(binding)

    def lookup(self, indices, out):
        self.lookup_native(indices, out)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        layout,
        layer_hash_index,
        use_sequence_parallel,
        prefix,
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
        # hash columns gathers only its own columns, side by side.
        columns = self.embed_tokens.columns
        self.register_buffer(
            "staged_rows",
            torch.zeros(
                caps.max_tokens,
                6144 if columns is None else len(columns) * 256,
                dtype=torch.bfloat16,
                device=caps.device,
            ),
            persistent=False,
        )
        # A resident table's lookup is prepared through B12X (the lookup unit).
        set_b12x_preparation_provider(self, self)

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if workload.max_tokens > self.embed_tokens.caps.max_tokens:
            return ()

        def make_call(state):
            from b12x.sequence.engram._impl import _bind_lookup_state

            embed = self.embed_tokens
            hash_ids = torch.full(
                (embed.caps.max_tokens, 24),
                -1,
                dtype=torch.int64,
                device=embed.caps.device,
            )
            prime_row = min(embed.shard_start + 1, embed.shard_end - 1)
            if prime_row <= 0:
                raise PreparationResourceUnavailableError(
                    "Engram shard has no nonzero local lookup row"
                )
            hash_ids[0, 0] = prime_row
            num_tokens = torch.ones((1,), dtype=torch.int32, device=embed.caps.device)
            out = torch.empty(
                (embed.caps.max_tokens, 6144),
                dtype=torch.bfloat16,
                device=embed.caps.device,
            )
            binding = _bind_lookup_state(
                state,
                weight=embed.weight,
                scales=embed.weight_scale_inv,
                hash_ids=hash_ids,
                num_tokens=num_tokens,
                out=out,
            )
            return PreparedCall(
                run=lambda: state.run_lookup(binding, 1, clear_tail=True),
                owners=(embed.weight, embed.weight_scale_inv),
            )

        if self.embed_tokens.table_memory == "disk":
            return ()
        request = self.embed_tokens.plan.request(
            name=f"{id(self)}/engram-lookup",
            prepare_call=make_call,
            benchmark_call=make_call,
        )
        return (
            B12xPreparationUnit(
                name="EngramLookup",
                key=(id(self), workload.max_tokens),
                requests=(request,),
                stage="weights",
            ),
        )

    def declare_kernels(self) -> None:
        """Declare the TileLang gate: on packed rows always, and on the column blocks
        of a column-parallel projection when the model runs prefill sequence
        parallel (the blocked gate, one block per rank). The model passes the
        step's int32 token ids, the gate finding the image sentinel itself."""
        from vllm.models.deepseek_v4_1.tilelang import prepare
        from vllm.models.deepseek_v4_1.tilelang.engram import (
            engram_gate_blocked_kernel,
            engram_gate_kernel,
        )

        calls = [
            prepare.call(engram_gate_kernel, self.hc_mult, self.dim, self.eps, "int32")
        ]
        sequence_parallel = (
            0 < self.prefill_sp_min_rows <= self.embed_tokens.caps.max_tokens
        )
        if sequence_parallel:
            # SP's trailing pad rows pass through (forward).
            from vllm.models.deepseek_v4_1.tilelang import columns

            key = columns.copy_columns_key(self.hc_mult * self.dim, torch.bfloat16)
            calls.append(prepare.call(columns.copy_columns_kernel, *key))
        if sequence_parallel and isinstance(self.wkv, _PaddedColumnParallelLinear):
            calls.append(
                prepare.call(
                    engram_gate_blocked_kernel,
                    self.hc_mult,
                    self.dim,
                    self.eps,
                    "int32",
                    get_tensor_model_parallel_world_size(),
                    self.wkv.output_size_per_partition,
                )
            )
        prepare.declare(lambda: calls)

    def process_weights_after_loading(self):
        self.norm_weights.copy_(
            (self.q_weight.float() * self.k_weight.float()).flatten()
        )
        self.embed_tokens.open_disk_rows()

    def stage_rows(self, tokens, stream, early=None):
        """Queue the disk rows of this step's ``tokens`` tokens (their row IDs in
        ``embed_tokens.hashes``; ``early``: the IDs the host hashed); the forward
        gathers them."""
        self.invalidate_disk_output()
        self.embed_tokens.stage_rows(tokens, stream, early)
        self._disk_prepared_tokens = tokens
        self._disk_prepared = True

    def prepare_embeddings(self, hash_ids):
        self.embed_tokens.lookup(hash_ids, self.staged_rows)

    def invalidate_disk_output(self, *, clear=False):
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        if clear:
            self.staged_rows.zero_()

    def prepare_dummy_output(self, num_tokens):
        self.embed_tokens.stage_no_rows()
        self._disk_prepared_tokens = num_tokens
        self._disk_prepared = True

    def forward(self, hidden_states, hash_ids, token_mask=None, sp_rows=None):
        """Gate ``hidden_states`` with this layer's Engram rows.

        ``token_mask`` is the step's token ids (the image sentinel's rows pass
        through).

        With ``sp_rows`` (prefill sequence parallelism) ``hidden_states`` holds
        this rank's rows while ``hash_ids`` and ``token_mask`` cover every
        token; the lookup and projection still run on every row, and with a
        column-parallel projection each rank receives only its own rows of the
        product (an all-to-all instead of an all-gather of every row).
        """
        if (
            self.embed_tokens.table_memory == "disk"
            and not torch.compiler.is_compiling()
            and (
                not self._disk_prepared
                or self._disk_prepared_tokens < hash_ids.shape[0]
            )
        ):
            raise RuntimeError("Disk Engram output is not prepared")
        if self.embed_tokens.table_memory == "disk":
            self.embed_tokens.gather_rows(self.staged_rows[: hash_ids.shape[0]])
        if self.embed_tokens.columns is not None:
            # This rank's gathered rows are its hash columns, every tp-th.
            rows = interleaved_rows(self.staged_rows[: hash_ids.shape[0]])
        else:
            rows = tensor_model_parallel_all_reduce(
                self.staged_rows[: hash_ids.shape[0]]
            )
        # With SP and a column-parallel projection, kv is every rank's column block
        # of this rank's rows, [N, L, C]: the gate reads the blocks in place.
        blocked = sp_rows is not None and isinstance(
            self.wkv, _PaddedColumnParallelLinear
        )
        if blocked:
            kv = sp_rows.exchange_rows(self.wkv.local_product(rows))
            kv_start = 0
        else:
            kv = self.wkv(rows)
            kv_start = 0 if sp_rows is None else sp_rows.start
        state = hidden_states.flatten(1)
        out = torch.empty_like(state)
        mix_out = out
        if sp_rows is not None:
            # Gate the real local rows; the trailing pad rows pass through.
            start, real = sp_rows.start, sp_rows.real_rows
            if real < state.shape[0]:
                from vllm.models.deepseek_v4_1.tilelang import columns

                columns.copy_columns(state[real:], out[real:])
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
