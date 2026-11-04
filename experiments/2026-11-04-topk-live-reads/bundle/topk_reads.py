"""Independent top-k scratch-read gate, with no clustered-search imports."""
import torch
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK

width, extent, k = 16384, 576, 512
scores = torch.empty((1, width), device='cuda', dtype=torch.bfloat16)
scores[:, :extent].fill_(1)
lengths = torch.tensor([extent], device='cuda', dtype=torch.int32)
blocks = torch.arange(width // 8, device='cuda', dtype=torch.int32).view(1, -1)
out = torch.empty((1, k), device='cuda', dtype=torch.int32)
values = torch.empty((1, k), device='cuda', dtype=torch.bfloat16)
IndexerTopK(width, k, candidates=True).run(scores, lengths, None, out, values, gather=blocks)
torch.cuda.synchronize()
torch.testing.assert_close(out, torch.arange(k, device='cuda', dtype=torch.int32).view(1, -1), atol=0, rtol=0)
print('exact top-k output; initcheck must also report zero unwritten scratch reads', flush=True)
