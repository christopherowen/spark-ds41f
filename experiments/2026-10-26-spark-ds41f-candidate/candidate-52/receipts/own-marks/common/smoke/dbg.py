import sys, importlib.util, torch
sys.path.insert(0, "/opt/spark-ds41f/candidate/vllm")
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bra=load("bra","/bench/bench_reference_alignment.py"); ref=load("refm","/ref/test_deepseek_v41_tilelang_moe.py")
sys.argv=["x","dbg"]
h1030=load("h1030","/h/h1030.py")
inter=576; layout,h=bra._swiglu_case(16,inter,384,6,block_M=16)
order,w,te,psum=layout["order"],layout["layout_weights"],layout["tile_expert"],layout["psum"]
live=order>=0
rq,rsf=bra._swiglu_outputs(h.shape[0],inter); ref._tile_kernels_swiglu(inter,384,16)(h,rq,rsf,psum,w,1.0,None)
tq,tsf=h1030.torch_ref(h,w,inter,floor=False)
print("rows",h.shape, "psum",psum.tolist()[-3:], "live",int(live.sum()), "w", w.dtype, w.shape)
cols=inter//32
d=(rsf[live,:cols].int()-tsf[live,:cols].int())
print("scale delta hist", torch.unique(d, return_counts=True))
print(rsf[live][:2,:cols].tolist()); print(tsf[live][:2,:cols].tolist())
print("rsf shape", rsf.shape, rsf.dtype)
tq1,tsf1=h1030.torch_ref(h,w,inter,floor=True)
d1=(rsf[live,:cols].int()-tsf1[live,:cols].int()); print("floor delta hist", torch.unique(d1, return_counts=True))
L=torch.nonzero(live).flatten()
r0=L[1].item()
a=rq[r0,:32].float(); b=tq[r0,:32].float()
print("row",r0,"sf",rsf[r0,0].item(),tsf[r0,0].item()); print(a.tolist()); print(b.tolist())
hf=h[r0].float(); g=hf[:inter].clamp(max=10.0); u=hf[inter:].clamp(-10.0,10.0)
x=(g/(1+torch.exp(-g))*u*w[r0]).bfloat16().float()
print("w",w[r0].item(),"x",x[:8].tolist())
import tile_kernels
print(tile_kernels.__file__)
print("tsf1", tsf1.shape, tsf1.dtype, tsf1[live][:2,:cols].tolist())
print("tsf", tsf.shape, tsf[live][:2,:cols].tolist())
bad=(rsf[:, :cols].int()!=tsf1[:, :cols].int()) & live[:,None]
rows_bad=torch.nonzero(bad.any(1)).flatten()
print("n_bad_rows", rows_bad.numel(), rows_bad[:10].tolist(), "n", int(psum[-1].item()))
r=rows_bad[0].item(); print("rsf", rsf[r,:cols].tolist(), "tsf1", tsf1[r,:cols].tolist(), "order", order[r].item(), "w", w[r].item(), "h0", h[r,:4].tolist())
q16=(rq[live].view(torch.uint8)!=tq1[live].view(torch.uint8))
vals=(rq[live].float()!=tq1[live].float())
print("byte diffs", int(q16.sum()), "value diffs", int(vals.sum()))
