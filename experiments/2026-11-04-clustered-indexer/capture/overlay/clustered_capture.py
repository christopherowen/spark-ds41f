# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot, explicitly armed real indexer snapshot for the cluster-search lab."""
import os
from pathlib import Path

import torch

_captured = set()


def maybe_capture(q, scales, weights, pool, table, lengths, active, width, page_size, candidate_blocks=False, context=None):
    directory = os.environ.get('VLLM_DS41_CLUSTER_CAPTURE')
    if not directory or q.shape[0] == 0 or not Path(directory, 'arm').is_file():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    kind = bool(candidate_blocks)
    key = (pool.data_ptr(), kind)
    budget = 2 if kind else 3
    ordinal = sum(item[1] == kind for item in _captured)
    if key in _captured or ordinal >= budget:
        return
    limit = max(min(int(active[0].item()), width), 0)
    row = int(lengths.argmax().item())
    extent = min(int(lengths[row].item()), limit)
    minimum = int(os.environ.get('VLLM_DS41_CLUSTER_CAPTURE_MIN_KEYS', '32769'))
    if extent < minimum:
        return
    # Copy only written byte prefixes. Reading a whole final page and clearing
    # its tail afterwards would still read unwritten device memory.
    logical_pages = (extent + page_size - 1) // page_size
    logical_ids = table[row, :logical_pages].detach().cpu().long()
    ids, inverse = torch.unique(logical_ids, sorted=True, return_inverse=True)
    from .indexer import page_bytes
    nbytes = page_bytes(page_size)
    byte_pool = pool.view(torch.uint8)
    prefixes = [0] * len(ids)
    for logical, compact in enumerate(inverse.tolist()):
        prefixes[compact] = max(prefixes[compact], min(page_size, extent - logical * page_size))
    selected = torch.zeros((len(ids), nbytes), dtype=torch.uint8)
    for compact, (physical, count) in enumerate(zip(ids.tolist(), prefixes)):
        selected[compact, :count * 64].copy_(byte_pool[physical, :count * 64].detach().cpu())
        start = page_size * 64
        selected[compact, start:start + count * 4].copy_(
            byte_pool[physical, start:start + count * 4].detach().cpu())
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    pool_ordinal = ordinal
    interface = 'source' if kind else 'dense'
    target = Path(directory, f'rank-{rank}-{interface}-{pool_ordinal}.pt')
    if target.exists():
        raise RuntimeError(f'refusing to overwrite a real-input capture: {target}')
    snapshot = {
        'q': q[row:row+1].detach().cpu(),
        'scales': scales[row:row+1].detach().cpu(),
        'weights': weights[row:row+1].detach().cpu(),
        'pool': selected,
        'table': inverse.to(torch.int32).view(1, -1),
        'lengths': torch.tensor([extent], dtype=torch.int32),
        'active': torch.tensor([limit], dtype=torch.int32),
        'page_size': page_size,
        'width': width,
        'candidate_blocks': bool(candidate_blocks),
        'schema_version': 2,
        'capture_context': context or {},
        'capture_minimum_keys': minimum,
        'layout': {
            'pool_pages': byte_pool.shape[0],
            'row_stride_bytes': byte_pool.stride(0),
            'alignment_mod_256': byte_pool.data_ptr() % 256,
            'query_rows': q.shape[0],
            'captured_row': row,
            'table_columns': table.shape[1],
            'physical_ids': ids,
            'written_prefixes': prefixes,
        },
        'rank': rank,
        'pool_ordinal': pool_ordinal,
        'source': 'candidate-36, tree cc0fdf582d0bddff71c6a853e237bc9e87587832',
    }
    # Atomic publication: a partial capture is never replayed as an input.
    temporary = target.with_suffix('.tmp')
    if temporary.exists():
        raise RuntimeError(f'refusing to overwrite a partial capture: {temporary}')
    torch.save(snapshot, temporary)
    temporary.replace(target)
    _captured.add(key)
    print(f'clustered indexer snapshot rank={rank} extent={extent} path={target}', flush=True)
