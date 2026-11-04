# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot, explicitly armed real indexer snapshot for the cluster-search lab."""
import os
from pathlib import Path

import torch

_done = False


def maybe_capture(q, scales, weights, pool, table, lengths, active, width, page_size):
    global _done
    directory = os.environ.get('VLLM_DS41_CLUSTER_CAPTURE')
    if _done or not directory or q.shape[0] == 0 or not Path(directory, 'arm').is_file():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    limit = min(int(active[0].item()), width)
    row = int(lengths.argmax().item())
    extent = min(int(lengths[row].item()), limit)
    if extent < 8192:
        return
    # Only one query and the cache pages it can read. No whole-pool host copy.
    logical_pages = (extent + page_size - 1) // page_size
    ids = table[row, :logical_pages].long()
    from .indexer import page_bytes
    nbytes = page_bytes(page_size)
    selected = pool[:, :nbytes].index_select(0, ids)
    tail = extent % page_size
    if tail:
        selected[-1, tail * 64:page_size * 64] = 0
        selected[-1, page_size * 64 + tail * 4:] = 0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    target = Path(directory, f'rank-{rank}.pt')
    snapshot = {
        'q': q[row:row+1].detach().cpu(),
        'scales': scales[row:row+1].detach().cpu(),
        'weights': weights[row:row+1].detach().cpu(),
        'pool': selected.detach().cpu(),
        'table': torch.arange(logical_pages, dtype=torch.int32).view(1, -1),
        'lengths': torch.tensor([extent], dtype=torch.int32),
        'active': torch.tensor([logical_pages * page_size], dtype=torch.int32),
        'page_size': page_size,
        'width': logical_pages * page_size,
        'rank': rank,
        'source': 'candidate-36, tree cc0fdf582d0bddff71c6a853e237bc9e87587832',
    }
    # Atomic publication: a partial capture is never replayed as an input.
    temporary = target.with_suffix('.tmp')
    torch.save(snapshot, temporary)
    temporary.replace(target)
    _done = True
    print(f'clustered indexer snapshot rank={rank} extent={extent} path={target}', flush=True)
