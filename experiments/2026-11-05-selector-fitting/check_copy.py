"""CPU-only initialization/table tests; native GPU replay is a separate gate."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from overlay import selector_capture as capture


def check(page):
    torch.manual_seed(page)
    rows,width = 65,65536
    extent = torch.arange(rows,dtype=torch.int32)+32769
    pages = (int(extent.max())+page-1)//page
    pool = torch.randint(0,256,(2*pages+5,page*68),dtype=torch.uint8)
    table = torch.arange(pages,dtype=torch.int32).expand(rows,pages).clone()
    table[1::2] += pages+2
    table[:,1] = table[:,0]  # Aliased physical page, with separate query tables.
    q = torch.randint(0,256,(rows,32,64),dtype=torch.uint8)
    scales = torch.randint(0,256,(rows,32,4),dtype=torch.uint8)
    weights = torch.randn(rows,32,dtype=torch.bfloat16)
    active = torch.tensor([width],dtype=torch.int32)
    args = (q,scales,weights,pool,table,extent,active,width,page)
    data = capture.copy_batch(*args)
    assert len(data['q']) == 64
    torch.testing.assert_close(data['q'],q[1:])
    for r,original in enumerate(range(1,rows)):
        count = (int(extent[original])+page-1)//page
        ids = data['layout']['physical_ids'][data['table'][r,:count].long()]
        torch.testing.assert_close(ids,table[original,:count].long())
    for dest,(physical,count) in enumerate(zip(data['layout']['physical_ids'],data['layout']['written_prefixes'])):
        torch.testing.assert_close(data['pool'][dest,:count*64],pool[physical,:count*64])
        torch.testing.assert_close(data['pool'][dest,page*64:page*64+count*4],pool[physical,page*64:page*64+count*4])
        assert not data['pool'][dest,count*64:page*64].count_nonzero()
        assert not data['pool'][dest,page*64+count*4:].count_nonzero()
    assert capture.copy_batch(*args,max_rows=64,min_keys=width) is None
    with tempfile.TemporaryDirectory() as directory,patch('torch.cuda.is_current_stream_capturing',return_value=False),patch('torch.distributed.is_initialized',return_value=False),patch.dict(os.environ,{'VLLM_DS41_SELECTOR_CAPTURE':directory}):
        capture._captured.clear()
        context = {'prefix':'synthetic.layer','layer_id':20,'mode':'prefill'}
        capture.maybe_capture(*args,True,context)
        assert not list(Path(directory).glob('*.pt'))
        arm = Path(directory,'arm.json')
        arm.write_text(json.dumps({'case_id':'synthetic-copy','prompt_sha256':'a'*64}))
        arm.write_text(json.dumps({'case_id':'synthetic-copy','prompt_sha256':'a'*64,'source_min_keys':width+1}))
        capture.maybe_capture(*args,True,context)
        assert not list(Path(directory).glob('*.pt'))
        arm.write_text(json.dumps({'case_id':'synthetic-copy','prompt_sha256':'a'*64}))
        with patch('torch.cuda.is_current_stream_capturing',return_value=True):
            capture.maybe_capture(*args,True,context)
            assert not list(Path(directory).glob('*.pt'))
        capture.maybe_capture(*args,True,dict(context,mode='decode'))
        assert not list(Path(directory).glob('*.pt'))
        capture.maybe_capture(*args,True,context)
        paths = list(Path(directory).glob('*.pt'));assert len(paths) == 1
        capture.maybe_capture(*args,True,context)
        assert len(list(Path(directory).glob('*.pt'))) == 1
        capture._captured.clear()
        try:
            capture.maybe_capture(*args,True,context)
        except AssertionError as error:
            assert 'overwrite' in str(error)
        else:
            raise AssertionError('capture overwrite was accepted')
        saved = torch.load(paths[0],weights_only=True)
        assert saved['mode'] == 'prefill' and saved['capture_context'] == context
        for name in ('q','scales','weights','pool','table','lengths','active'):
            torch.testing.assert_close(saved[name],data[name])
    return {'page_size':page,'rows':64,'distinct_query_tables':True,'aliased_pages':True,
            'written_prefix_only':True,'zero_saved_tails':True,'unarmed_and_graph_skip':True,
            'decode_skip':True,'overwrite_refused':True,'cpu_only':True}


if __name__ == '__main__':
    print(json.dumps({'passed':True,'cases':[check(128),check(256)]}))
