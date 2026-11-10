# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly armed batches of native operands for separate selector fitting.

Diagnostic profile only. Host copies and synchronization never run in the timed
LSA profile. No capture changes native selection, attention or model weights.
"""
import json
import os
from pathlib import Path
import re

import torch

_captured = set()


def copy_batch(q,scales,weights,pool,table,lengths,active,width,page_size,*,max_rows=64,min_keys=32769):
    """Copy eligible rows and the union of their written physical page prefixes.

    Tables may differ between queries. Only visible table entries and initialized
    native key/scale prefixes are read. Unwritten saved tails are zero-filled.
    """
    limit = max(min(int(active[0].item()),width),0)
    extents = lengths.detach().cpu().long().clamp(min=0,max=limit)
    rows = torch.nonzero(extents >= min_keys).flatten()[-max_rows:]
    if len(rows) < max_rows:
        return None
    extent = extents[rows]
    counts = (extent+page_size-1)//page_size
    logical = [table[int(row),:int(count)].detach().cpu().long() for row,count in zip(rows,counts)]
    ids = torch.unique(torch.cat(logical),sorted=True)
    byte_pool = pool.view(torch.uint8)
    assert bool(((ids >= 0) & (ids < len(byte_pool))).all()), 'invalid physical page id'
    nbytes = page_size*68
    assert len(ids)*nbytes <= 512*1024**2, 'capture union exceeds 512 MiB host budget'
    index = {physical:i for i,physical in enumerate(ids.tolist())}
    compact = torch.zeros((len(rows),int(counts.max())),dtype=torch.int32)
    prefixes = [0]*len(ids)
    for r,(physical_ids,nkeys) in enumerate(zip(logical,extent.tolist())):
        for col,physical in enumerate(physical_ids.tolist()):
            dest = index[physical]
            compact[r,col] = dest
            prefixes[dest] = max(prefixes[dest],min(page_size,nkeys-col*page_size))
    saved = torch.zeros((len(ids),nbytes),dtype=torch.uint8)
    for dest,(physical,count) in enumerate(zip(ids.tolist(),prefixes)):
        saved[dest,:count*64].copy_(byte_pool[physical,:count*64].detach().cpu())
        start = page_size*64
        saved[dest,start:start+count*4].copy_(byte_pool[physical,start:start+count*4].detach().cpu())
    device_rows = rows.to(q.device)
    return {'schema_version':3,'q':q[device_rows].detach().cpu(),
            'scales':scales[device_rows].detach().cpu(),'weights':weights[device_rows].detach().cpu(),
            'pool':saved,'table':compact,'lengths':extent.int(),
            'active':torch.tensor([limit],dtype=torch.int32),'page_size':page_size,'width':width,
            'layout':{'physical_ids':ids,'written_prefixes':prefixes,'captured_rows':rows,
                      'query_rows':len(q),'table_columns':table.shape[1],
                      'pool_pages':len(byte_pool),'row_stride_bytes':byte_pool.stride(0),
                      'alignment_mod_256':byte_pool.data_ptr()%256}}


def maybe_capture(q,scales,weights,pool,table,lengths,active,width,page_size,candidate_blocks=False,context=None):
    directory = os.environ.get('VLLM_DS41_SELECTOR_CAPTURE')
    if not directory or not q.shape[0] or torch.cuda.is_current_stream_capturing():
        return
    arm = Path(directory,'arm.json')
    if not arm.is_file():
        return
    request = json.loads(arm.read_text())
    case = request['case_id']
    assert re.fullmatch('[a-z0-9-]{1,64}',case)
    assert re.fullmatch('[a-f0-9]{64}',request['prompt_sha256'])
    context = context or {}
    assert isinstance(context.get('layer_id'),int) and context.get('prefix'), 'explicit native caller identity required'
    if context.get('mode') != 'prefill':
        return
    kind = bool(candidate_blocks)
    key = (case,pool.data_ptr(),kind)
    ordinal = sum(k[0] == case and k[2] == kind for k in _captured)
    if key in _captured or ordinal >= (1 if kind else 3):
        return
    minimum = request.get('source_min_keys' if kind else 'dense_min_keys',32769)
    assert isinstance(minimum,int) and minimum > 0
    data = copy_batch(q,scales,weights,pool,table,lengths,active,width,page_size,min_keys=minimum)
    if data is None:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    interface = 'source' if kind else 'dense'
    target = Path(directory,f'{case}-rank-{rank}-{interface}-{ordinal}.pt')
    temporary = target.with_suffix('.tmp')
    assert not target.exists() and not temporary.exists(), 'refusing capture overwrite'
    data.update(case_id=case,prompt_sha256=request['prompt_sha256'],capture_context=context,
                candidate_blocks=kind,rank=rank,pool_ordinal=ordinal,mode='prefill',
                capture_minimum_keys=minimum,
                source='native C36 with live-list scratch guard; diagnostic copies only')
    torch.save(data,temporary)
    temporary.replace(target)
    _captured.add(key)
    print(f'selector fitting capture rank={rank} layer={context["layer_id"]} rows={len(data["q"])} path={target}',flush=True)
