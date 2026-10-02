"""Bounded 64 GiB allocation, probing every 4 GiB DMA-submap boundary."""
import json,os,time
from pathlib import Path
import torch
assert os.sysconf('SC_PAGE_SIZE') == 65536
available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert available > 90*1024**3, available
start=time.monotonic()
x=torch.empty(64*1024**3,device='cuda',dtype=torch.uint8)
checks=[]
for boundary in range(4,64,4):
 for delta in (-131072,-65536,0,65536):
  offset=boundary*1024**3+delta
  value=(boundary+delta//65536)%251+1
  window=x[offset:offset+65536]
  window.fill_(value)
  torch.cuda.synchronize()
  assert bool(torch.all(window.cpu()==value)),offset
  checks.append(offset)
print(json.dumps({'passed':True,'allocation_bytes':x.numel(),'checked_offsets':checks,'seconds':time.monotonic()-start}))
