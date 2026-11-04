#!/usr/bin/env python3
"""Trace native-id score, selection and rank kernels in eager and graph execution."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

import torch  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang import iket  # noqa: E402

iket.enable(args.out, ctas=1, iters=1, warps=1, granularity="phase")
from vllm.v1.worker.gpu.sample import logprob  # noqa: E402


def eager_and_graph(call, check):
    check(call())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    check(output)


cases = []
for dtype in (torch.bfloat16, torch.float32):
    logits = torch.randn(3, 257, device="cuda", dtype=dtype)
    for id_dtype in (torch.int32, torch.int64):
        ids = torch.randint(257, (3, 42), device="cuda", dtype=id_dtype)[:, ::2]
        for masked in (False, True):
            valid = (ids % 3 != 0) if masked else None
            expected = logits.double().log_softmax(-1).gather(1, ids.long()).float()
            if masked:
                expected.masked_fill_(~valid, -torch.inf)

            def check_scores(output):
                torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-6)

            eager_and_graph(
                lambda: logprob.compute_token_logprobs(logits, ids, valid), check_scores
            )
        sampled = ids[:, 0]

        def check_topk(output):
            assert torch.equal(output.logprob_token_ids[:, 0], sampled)
            ref = (
                logits.double()
                .log_softmax(-1)
                .gather(1, output.logprob_token_ids)
                .float()
            )
            torch.testing.assert_close(output.logprobs, ref, rtol=2e-6, atol=1e-6)
            assert torch.equal(
                output.selected_token_ranks,
                (logits >= logits.gather(1, sampled[:, None].long())).sum(-1),
            )

        eager_and_graph(
            lambda: logprob.compute_topk_scores(logits, 5, sampled), check_topk
        )
        cases.append(dict(dtype=str(dtype), id_dtype=str(id_dtype), masked_variants=2))
torch.cuda.synchronize()
(args.out / "numerics.json").write_text(json.dumps(cases, indent=2) + "\n")
