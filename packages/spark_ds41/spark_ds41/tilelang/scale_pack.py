# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack UE8M0 block scales into the words an MXFP8 GEMM loads.

This module is torch only, so a CPU test can import it without TileLang.
"""

import torch


def pack_scale_words(exponents: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    """Pack ``[..., K / 32]`` UE8M0 bytes into ``[..., ceil(K / 128)]`` uint32 words.

    With ``rows``, a ``[rows / 32, K / 32]`` block-scale grid is first
    broadcast to every row. The MXFP8 GEMM stores the grid without that
    broadcast and loads ``SFB[row // 32]``; those two tensors satisfy
    ``expanded[row] == block[row // 32]``. Rows whose exponent count is not a
    multiple of four gain unread padding bytes (TP4's 576-wide rows: 18 bytes
    become 20).
    """
    exponents = exponents.view(torch.uint8)
    if rows is not None and exponents.shape[0] != rows:
        exponents = exponents.repeat_interleave(rows // exponents.shape[0], dim=0)
    pad = -exponents.shape[-1] % 4
    if pad:
        exponents = torch.nn.functional.pad(exponents, (0, pad), value=127)
    return exponents.contiguous().view(torch.uint32)
