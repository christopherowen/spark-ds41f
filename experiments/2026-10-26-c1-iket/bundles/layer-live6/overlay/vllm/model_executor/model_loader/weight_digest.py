# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A SHA-256 of the loaded parameters: the check that two loaders load the same bits.

With ``SPARK_DS41F_WEIGHT_DIGEST=1`` every rank logs, once its loader has filled the
parameters and before post-load processing turns them into kernel layouts::

    weight digest rank <rank> <sha256> <model class>: <n> parameters, <GB> GB in <s> s

The hash covers every parameter's bytes in name order, a shared parameter once under
its first name, so a rank whose digest is equal across loaders and boots holds the
same weights. A speculative model loads through the same loader and logs its own line.
The bytes pass through one 64 MiB host buffer and the CPU hashes them at a few GB/s,
so a DS4.1 rank starts some tens of seconds later. Production leaves the variable
unset, which adds no work at all.
"""

import hashlib
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("SPARK_DS41F_WEIGHT_DIGEST", "0") == "1"
_STAGING_BYTES = 64 << 20


def weight_digest(
    model: torch.nn.Module, staging_bytes: int = _STAGING_BYTES
) -> tuple[str, int, int]:
    """The SHA-256 of ``model``'s parameter bytes in name order, with the number of
    parameters and of bytes hashed."""
    digest = hashlib.sha256()
    # Touched before the first copy, so no copy writes into unfaulted host pages.
    staging = torch.zeros(staging_bytes, dtype=torch.uint8)
    count = total = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        tensor = parameter.detach()
        if tensor.is_meta:
            raise RuntimeError(f"weight digest: {name} has no storage after loading")
        rows = tensor.reshape(1) if tensor.dim() == 0 else tensor
        row_bytes = rows[0].numel() * rows.element_size() if rows.shape[0] else 0
        step = max(1, staging_bytes // max(row_bytes, 1))
        for start in range(0, rows.shape[0], step):
            # A view of a contiguous parameter; at most one staging buffer (or one
            # row) of temporary device memory for a strided one.
            chunk = rows[start : start + step].contiguous().view(-1).view(torch.uint8)
            for offset in range(0, chunk.numel(), staging_bytes):
                part = chunk[offset : offset + staging_bytes]
                host = staging[: part.numel()]
                host.copy_(part)
                digest.update(host.numpy())
        count += 1
        total += tensor.numel() * tensor.element_size()
    return digest.hexdigest(), count, total


def log_weight_digest(model: torch.nn.Module) -> None:
    started = time.perf_counter()
    sha256, count, total = weight_digest(model)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info(
        "weight digest rank %d %s %s: %d parameters, %.2f GB in %.1f s",
        rank,
        sha256,
        type(model).__name__,
        count,
        total / 1e9,
        time.perf_counter() - started,
    )
