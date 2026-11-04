"""Out-of-band snapshot metadata builder; its full cost is reported separately.

This is required index construction, not a correctness check. Serving must
eventually maintain it on cache writes and include that cost in request timing.
"""
import time
import torch


def decode(packed, scales):
    packed,scales = packed.view(torch.uint8),scales.view(torch.uint8)
    codes = torch.stack((packed & 15,packed >> 4),dim=-1).reshape(*packed.shape[:-1],128).long()
    values = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=packed.device)
    return values[codes]*torch.exp2(scales.float().repeat_interleave(32,dim=-1)-127)


def build(data):
    torch.cuda.synchronize()
    start = time.perf_counter()
    extent,page,width = int(data['lengths'][0]),int(data['page_size']),int(data['width'])
    complete = extent//512*512
    positions = torch.arange(complete,device='cuda')
    table,pool = data['table'].cuda(),data['pool'].cuda()
    physical,offset = table[0,positions//page].long(),positions%page
    packed = pool[physical[:,None],offset[:,None]*64+torch.arange(64,device='cuda')]
    scales = pool[physical[:,None],page*64+offset[:,None]*4+torch.arange(4,device='cuda')]
    points = decode(packed,scales).reshape(-1,64,8,128).mean(2)
    features = points/points.norm(dim=-1,keepdim=True).clamp(min=2**-50)
    chunks = len(points)
    ids = torch.arange(64,device='cuda').expand(chunks,64).clone()
    # Three balanced cosine splits produce eight buckets of eight native blocks.
    for level in range(3):
        group_count = 1 << level
        current = ids.reshape(chunks*group_count,-1)
        chunk_ids = torch.arange(chunks,device='cuda').repeat_interleave(group_count)
        x = features[chunk_ids[:,None],current]
        mean = x.mean(1)
        rows = torch.arange(len(current),device='cuda')
        a = x[rows,(x-mean[:,None]).square().sum(-1).argmax(1)]
        b = x[rows,(x-a[:,None]).square().sum(-1).argmax(1)]
        for iteration in range(4):
            order = (x*(a-b)[:,None]).sum(-1).argsort(dim=1,stable=True)
            half = x.shape[1]//2
            a = x.gather(1,order[:,:half,None].expand(-1,-1,128)).mean(1)
            b = x.gather(1,order[:,half:,None].expand(-1,-1,128)).mean(1)
        ids = current.gather(1,order).reshape(chunks,64)
    bucket_ids = ids.reshape(chunks,8,8)
    centers = torch.zeros((width//64,128),dtype=torch.bfloat16,device='cuda')
    membership = torch.full((width//8,),-1,dtype=torch.int32,device='cuda')
    if chunks:
        values = points[torch.arange(chunks,device='cuda')[:,None,None],bucket_ids].mean(2)
        centers[:chunks*8] = values.reshape(-1,128).to(torch.bfloat16)
        native_blocks = bucket_ids+torch.arange(chunks,device='cuda')[:,None,None]*64
        cluster = torch.arange(chunks*8,device='cuda').reshape(chunks,8,1).expand(-1,-1,8)
        membership.scatter_(0,native_blocks.flatten(),cluster.flatten().int())
    torch.cuda.synchronize()
    return centers,membership,{'construction_wall_ms':(time.perf_counter()-start)*1000,
        'initialized_keys':complete,'owner_scope':'one explicit immutable snapshot',
        'persistent_bytes':centers.numel()*centers.element_size()+membership.numel()*membership.element_size(),
        'chunks':chunks,'buckets':chunks*8,'native_blocks_per_bucket':8,
        'construction_in_query_timer':False,'construction_is_required_runtime_work':True}
