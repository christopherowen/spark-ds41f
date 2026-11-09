import torch
xb=torch.tensor([[[1.0,-3.0],[0.5,0.25]]],dtype=torch.float32,device="cuda")
for floor in (False,True):
    amax=xb.abs().amax(-1)
    if floor: amax=amax.clamp(min=1e-4)
    t=amax*torch.tensor(1.0/448.0,dtype=torch.float32,device="cuda")
    bits=t.view(torch.int32)
    print(floor, amax.dtype, amax.is_contiguous(), t.dtype, t.tolist(), bits.tolist(), ((bits>>23)&0xFF).tolist())
