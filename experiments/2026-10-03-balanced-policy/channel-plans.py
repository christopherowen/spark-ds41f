"""Read actual NCCL work partitions; distinguish payload from RDMA acknowledgements."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('log',type=Path);a=p.parse_args()
parts=None;plans={};rings={};roots={}
for line in a.log.read_text().splitlines():
 m=re.search(r'Ring (\d+) : (\d+) -> (\d+) -> (\d+)',line)
 if m:
  ch,prev,rank,nxt=map(int,m.groups());rings[ch]={'previous':prev,'rank':rank,'next':nxt}
 m=re.search(r'Channel (\d+)/\d+ : .*\[send\] via NET/IB/(\d+)',line)
 if m:roots[int(m[1])]=int(m[2])
 m=re.search(r'Channel partition: elementBytes=(\d+) countLo=(\d+) countMid=(\d+) countHi=(\d+)$',line)
 if 'Channel partition:' in line:parts=list(map(int,m.groups())) if m else None
 m=re.search(r'(AllReduce|AllGather|ReduceScatter|Broadcast): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo..Hi\}=\{(\d+)..(\d+)\}',line)
 if not m:continue
 op,bs,algo,proto,lo,hi=m.groups();bs,lo,hi=map(int,[bs,lo,hi])
 if parts is None:continue
 elem,c0,cm,c1=parts;parts=None
 counts=[c0] if lo==hi else [c0]+[cm]*(hi-lo-1)+[c1]
 if sum(counts)*elem!=bs:raise ValueError(f'partition/log mismatch: {line}')
 key=(op,bs,algo,proto,lo,hi,tuple(counts))
 plans[key]={'operation':op,'bytes_per_rank':bs,'algorithm':algo,'protocol':proto,'channels':{ch:{'payload_bytes':count*elem,'ring':rings[ch],'virtual_nic':roots[ch]} for ch,count in zip(range(lo,hi+1),counts)}}
print(json.dumps(list(plans.values()),indent=2))
