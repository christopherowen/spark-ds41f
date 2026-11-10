"""Native GPU replay of every captured row; no timing or quality claims."""
import json
import argparse
import hashlib
from pathlib import Path

import torch


def restore(data,max_bytes=8*2**30):
    assert data['schema_version'] == 3
    layout = data['layout']
    pages,stride = int(layout['pool_pages']),int(layout['row_stride_bytes'])
    nbytes = data['pool'].shape[1]
    required = (pages-1)*stride+nbytes+255
    assert pages > 0 and stride >= nbytes and required <= max_bytes
    backing = torch.empty(required,dtype=torch.uint8,device='cuda')
    offset = (int(layout['alignment_mod_256'])-backing.data_ptr())%256
    pool = backing.as_strided((pages,nbytes),(stride,1),offset)
    physical = layout['physical_ids'].long()
    pool.index_copy_(0,physical.cuda(),data['pool'].cuda())
    table = physical[data['table'].long()].int().cuda()
    assert pool.stride(0) == stride and pool.data_ptr()%256 == int(layout['alignment_mod_256'])
    args = [data[k].cuda() for k in ('q','scales','weights')]
    args.extend([pool,table,data['lengths'].cuda(),data['active'].cuda()])
    return args


def compare(data,original,*,original_rows=True):
    from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS
    rows = len(data['q'])
    selected = data['layout']['captured_rows'].long().cuda()
    native = [a[selected] if i in (0,1,2,4,5) else a for i,a in enumerate(original)] if original_rows else original
    restored = restore(data)
    results = []
    for source in (False,True):
        plan = IndexerV41Plan(data['width'],data['page_size'],candidate_blocks=source)
        outputs = []
        for args in (native,restored):
            options = {}
            if source:
                options = {'candidate_out':torch.empty((rows,CANDIDATE_BLOCKS),dtype=torch.int32,device='cuda'),
                           'candidate_out_lengths':torch.empty(rows,dtype=torch.int32,device='cuda')}
            out,scores = plan.run(*args,**options)
            outputs.append((out,scores,options))
        left,right = outputs
        torch.testing.assert_close(left[0],right[0],atol=0,rtol=0)
        for r,extent in enumerate(data['lengths'].tolist()):
            torch.testing.assert_close(left[1][r,:extent].view(torch.int16),right[1][r,:extent].view(torch.int16),atol=0,rtol=0)
        if source:
            for name in ('candidate_out','candidate_out_lengths'):
                torch.testing.assert_close(left[2][name],right[2][name],atol=0,rtol=0)
        results.append({'source_interface':source,'all_rows_native_keys_equal':True,
                        'visible_bf16_score_bits_equal':True,'native_source_handoff_equal':source})
    return results


def fixture(page):
    from vllm.models.deepseek_v4_1.tilelang.selector_capture import copy_batch
    rows,width = 65,65536
    g = torch.Generator(device='cuda').manual_seed(page)
    extents = torch.arange(rows,dtype=torch.int32,device='cuda')+32769
    logical = width//page
    # Two different query tables and an unused physical gap; preserve stride and
    # alignment while leaving every non-visible address unwritten.
    pool = torch.empty((2*logical+3,page*68+64),dtype=torch.uint8,device='cuda')
    table = torch.arange(logical,dtype=torch.int32,device='cuda').expand(rows,logical).clone()
    table[1::2] += logical+2
    table[:,1] = table[:,0]
    visible = torch.arange(int(extents.max()),device='cuda')
    for row in (0,1):
        physical,offset = table[row,visible//page].long(),visible%page
        keys = torch.randint(0,256,(len(visible),64),dtype=torch.uint8,device='cuda',generator=g)
        scales = torch.randint(120,130,(len(visible),4),dtype=torch.uint8,device='cuda',generator=g)
        for d in range(64):pool[physical,offset*64+d] = keys[:,d]
        for d in range(4):pool[physical,page*64+offset*4+d] = scales[:,d]
    q = torch.randint(0,256,(rows,32,64),dtype=torch.uint8,device='cuda',generator=g)
    scales = torch.randint(120,130,(rows,32,4),dtype=torch.uint8,device='cuda',generator=g)
    weights = (torch.randn(rows,32,device='cuda',generator=g)/64).bfloat16()
    args = [q,scales,weights,pool,table,extents,torch.tensor([width],dtype=torch.int32,device='cuda')]
    data = copy_batch(*args,width,page)
    assert len(data['q']) == 64
    return {'page_size':page,'rows':64,'distinct_tables':True,'aliased_pages':True,
            'native_stride_bytes':pool.stride(0),'results':compare(data,args)}


def actual(inputs,collection,out):
    assert not out.exists()
    manifest_path = collection/'snapshots.json'
    source_path = collection/'source.json'
    manifest = json.loads(manifest_path.read_text())
    expected = {name:row for files in manifest.values() for name,row in files.items() if '-rank-0-' in name}
    paths = sorted(inputs.glob('*-rank-0-*.pt'))
    assert set(p.name for p in paths) == set(expected) and len(paths) == 12
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'passed':False,'actual_captures':True,'performance_measured':False,
              'collection_manifest_sha256':digest(manifest_path),'collection_source_sha256':digest(source_path),
              'collection_commit':json.loads(source_path.read_text())['commit'],'files':{}}
    for path in paths:
        sha = digest(path)
        assert sha == expected[path.name]['sha256'] and path.stat().st_size == expected[path.name]['bytes']
        data = torch.load(path,map_location='cpu',weights_only=True)
        assert data['schema_version'] == 3 and data['rank'] == 0 and len(data['q']) == 64
        assert data['mode'] == 'prefill' and data['case_id'] in ('fit-a','fit-b','validation-a')
        compact = [data[k].cuda() for k in ('q','scales','weights','pool','table','lengths','active')]
        result = compare(data,compact,original_rows=False)
        report['files'][path.name] = {'sha256':sha,'bytes':path.stat().st_size,'rows':64,
            'layer':data['capture_context']['layer_id'],'case_id':data['case_id'],'results':result}
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'file':path.name,'all_rows_native_replay_equal':True}),flush=True)
    report['passed'] = True
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--collection',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.inputs:
        assert args.collection and args.out
        result = actual(args.inputs,args.collection,args.out)
    else:
        cases = [fixture(128),fixture(256)]
        result = {'passed':True,'synthetic':True,'performance_measured':False,'cases':cases}
    print('RESULT '+json.dumps(result),flush=True)
