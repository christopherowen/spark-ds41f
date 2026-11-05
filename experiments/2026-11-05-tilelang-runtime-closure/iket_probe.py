"""Numerically check each replacement family under eager and graph IKET capture."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

import torch
from vllm.models.deepseek_v4_1.tilelang import iket

iket.enable(args.out / "sources", ctas=1, iters=1, warps=1, granularity="phase")
from vllm.models.deepseek_v4_1.tilelang import collectives, gather
from vllm.v1.worker.gpu.sample import logprob, tilelang_sampling
from vllm.v1.worker.gpu.sample.prompt_logprob import compute_prompt_logprobs_with_chunking

torch.manual_seed(20261105)
checked = []


def exercise(label, launch, expected):
    values = expected if isinstance(expected, tuple) else (expected,)

    def check(actual):
        outputs = actual if isinstance(actual, tuple) else (actual,)
        assert len(outputs) == len(values)
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for a, b in zip(outputs, values, strict=True)), label

    check(launch())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = launch()
    graph.replay()
    check(actual)
    checked.append(label)


for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for layout in ("padded", "sliced"):
        width = 1028 if layout == "padded" else 43
        backing = torch.randn((5, 1056 if layout == "padded" else 86), device="cuda", dtype=dtype)
        source = backing[:, :width] if layout == "padded" else backing[:, ::2]
        exercise(f"copy-{dtype}-{layout}", lambda: tilelang_sampling.copy_logits(source), source.float())
    sources = [torch.randn((5, 25), device="cuda", dtype=dtype)[:, :17] for _ in range(3)]
    expected = sources[0].float().clone()
    for source in sources[1:]:
        expected.add_(source.float())
    exercise(f"rank-strided-{dtype}", lambda: collectives.rank_order_sum(sources), expected.to(dtype))

source = torch.randn((32, 4112), device="cuda")[:, :4100]
exercise("copy-padded32", lambda: tilelang_sampling.copy_logits(source), source.clone())
source = torch.randn((32, 129296), device="cuda")[:, :129280]
exercise("copy-pitched-dma", lambda: tilelang_sampling.copy_logits(source), source.clone())
sources = [torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
expected = sources[0].float().clone()
for source in sources[1:]:
    expected.add_(source.float())
exercise("rank-packed", lambda: collectives.rank_order_sum(sources), expected.bfloat16())

for world, parts, local, width in ((3, 1, 16, 45), (4, 2, 16, 64)):
    source = torch.randn((world * parts * 5, local), device="cuda")
    outputs = tuple(torch.empty((5, width), device="cuda") for _ in range(parts))
    expected = tuple(source.view(world, parts, 5, local).permute(1, 2, 0, 3)
                     .reshape(parts, 5, world * local)[p, :, :width].contiguous()
                     for p in range(parts))

    def launch():
        gather.gather_columns_into(source, outputs, world)
        return outputs

    exercise(f"gather-{world}-{parts}", launch, expected)

for count in (7, 129):
    for dtype in (torch.uint8, torch.bfloat16, torch.float32):
        source = torch.arange(3 * count * 5, device="cuda").to(dtype).view(3 * count, 5)
        expected = source.view(3, count, 5).movedim(0, 1).reshape(count, 15)
        exercise(f"gather-tail-{count}-{dtype}",
                 lambda: collectives.concat_rank_shards(source, 3, (count, 5)), expected)

for id_dtype in (torch.int32, torch.int64):
    logits = torch.randn((5, 264), device="cuda")[:, :257]
    ids = torch.randint(257, (10,), device="cuda", dtype=id_dtype)[::2]
    for k in (0, 5):
        for mode in ("raw_logits", "raw_logprobs"):
            reference = logprob.compute_topk_scores(logits, k, ids, logits_mode=mode == "raw_logits")
            expected = tuple(reference)[:3]
            exercise(f"prompt-{id_dtype}-{k}-{mode}",
                     lambda: compute_prompt_logprobs_with_chunking(ids, logits, lambda hidden: hidden, k, mode),
                     expected)

torch.cuda.synchronize()
(args.out / "numerics.json").write_text(json.dumps(dict(cases=checked, count=len(checked)), indent=2) + "\n")
