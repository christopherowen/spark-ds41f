import torch, time
p = torch.cuda.get_device_properties(0)
print("name", p.name, "total MiB", p.total_memory/2**20, "integrated", getattr(p, "is_integrated", None))
vals=[]
for i in range(16):
    free, total = torch.cuda.mem_get_info()
    vals.append(free)
    time.sleep(0.2)
print("free samples MiB:", [round(v/2**20,1) for v in vals])
print("spread MiB:", round((max(vals)-min(vals))/2**20,1))
x = torch.empty(64<<20, dtype=torch.uint8, device="cuda"); x.fill_(1)
torch.cuda.synchronize()
free2,_ = torch.cuda.mem_get_info()
print("after 64MiB torch alloc: free drop MiB", round((vals[-1]-free2)/2**20,1), "reserved MiB", torch.cuda.memory_reserved()/2**20)
nt = lambda: (torch.cuda.synchronize(), torch.cuda.mem_get_info())[1]
a = [total - f - torch.cuda.memory_reserved() for f in [nt()[0] for _ in range(8)]]
print("non-torch MiB samples:", [round(v/2**20,1) for v in a], "spread", round((max(a)-min(a))/2**20,1))
