# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEBUG ONLY: per-row hashes of the attention's stages, to find where a request's
rows first differ between two batches.

With ``VLLM_DS41_ROW_TRACE=<dir>``, TP rank 0 hashes every row of each attention
layer's input, attention output, WO partial (before the TP sum) and WO result (after
it), with the row's position, in eager forwards. A forward ends when the first
target layer runs again; forwards with a prefill row are pickled (plain lists) as
``<dir>/step-<n>.pkl``. Hashes are a fixed weighted sum of the row's 32-bit words.
"""

import os
import pickle

import torch

DIR = os.environ.get("VLLM_DS41_ROW_TRACE", "")
_state = {"first": None, "records": [], "prefill": False, "step": 0}
_weights: dict = {}


def enabled() -> bool:
    if not DIR or torch.cuda.is_current_stream_capturing():
        return False
    from vllm.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank() == 0


def _hash(x: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    words = x.reshape(rows, -1).contiguous().view(torch.int32)
    key = (words.shape[1], words.device)
    if key not in _weights:
        gen = torch.Generator().manual_seed(words.shape[1])
        _weights[key] = torch.randint(
            1, 1 << 16, (words.shape[1],), generator=gen
        ).to(device=words.device, dtype=torch.int64)
    out = torch.empty(rows, dtype=torch.int64, device=words.device)
    for lo in range(0, rows, 256):
        out[lo : lo + 256] = (words[lo : lo + 256].to(torch.int64) * _weights[key]).sum(-1)
    return out


def begin(prefix: str, is_draft: bool) -> None:
    """Start a layer; the first target layer seen again closes the forward."""
    if _state["first"] is None and not is_draft:
        _state["first"] = prefix
    if prefix == _state["first"]:
        flush()


def record(prefix: str, stage: str, positions: torch.Tensor, x: torch.Tensor, **meta) -> None:
    if x is None or x.shape[0] == 0:
        return
    if meta.get("prefill"):
        _state["prefill"] = True
    _state["records"].append(
        (prefix, stage, positions[: x.shape[0]].detach().clone(), _hash(x), meta)
    )


def flush() -> None:
    records, prefill = _state["records"], _state["prefill"]
    _state["records"], _state["prefill"] = [], False
    if not records or not prefill:
        return
    os.makedirs(DIR, exist_ok=True)
    rows = [
        (prefix, stage, positions.cpu().tolist(), hashes.cpu().tolist(), meta)
        for prefix, stage, positions, hashes, meta in records
    ]
    with open(os.path.join(DIR, f"step-{_state['step']:06d}.pkl"), "wb") as f:
        pickle.dump(rows, f)
    _state["step"] += 1
