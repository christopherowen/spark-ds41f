# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""O_DIRECT checkpoint reads into ordinary CUDA tensors, for ``--load-format direct``
(``direct_loader``). Importing this package neither initializes CUDA nor builds the
native helper."""

import operator

from vllm.model_executor.model_loader.direct_io._checkpoint import (
    DirectWeightSession,
    TensorRange,
    scratch_chunks,
    view_offset,
    view_records,
)


def capabilities(device: int = 0) -> dict[str, int]:
    """The CUDA device attributes that decide whether the direct loader can run on
    ``device`` (``cudaDeviceGetAttribute``)."""
    from vllm.model_executor.model_loader.direct_io._native import load

    return load().capabilities(operator.index(device))


__all__ = [
    "DirectWeightSession",
    "TensorRange",
    "capabilities",
    "scratch_chunks",
    "view_offset",
    "view_records",
]
