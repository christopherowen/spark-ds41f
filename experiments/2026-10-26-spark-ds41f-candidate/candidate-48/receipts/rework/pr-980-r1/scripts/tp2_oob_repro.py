#!/usr/bin/env python3
"""Deterministic reproduction of the TP2 screen's illegal memory access.

rejection_sample reads draft_sampled at logits rows (start + i + 1), so it needs
[num_logits] = num_reqs * (1 + num_steps) entries; PR #980's screen passes
draft_tokens.view(-1), num_reqs * num_steps entries. The last request's reads run
num_reqs elements past the tensor, into whatever its allocator block holds.

This runs the screen's TP2 shapes (8 requests, 5 steps, vocab 8192, temperature
0.8, unseeded) in one process per mode, with no TP and none of #980's code on the
call path (rejection_sample is not in #980's files):
  rows            draft_sampled laid out as [num_logits], as the worker passes it
  short_zero      the screen's 40-entry argument, the 8 entries past it zero
  short_stale     the screen's 40-entry argument, the 8 entries past it holding
                  an out-of-vocabulary value, as stale allocator memory can
"""

import json
import subprocess
import sys


def run(mode: str) -> None:
    import torch

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    num_reqs, num_steps, vocab = 8, 5, 4096 * 2
    rows = 1 + num_steps
    num_logits = num_reqs * rows
    g = torch.Generator(device=device).manual_seed(7)
    target = torch.randn((num_logits, vocab), generator=g, device=device)
    draft_logits = torch.randn((num_reqs, num_steps, vocab), generator=g, device=device)
    drafts = torch.randint(0, vocab, (num_reqs, num_steps), generator=g, device=device)
    if mode == "rows":
        draft_sampled = torch.zeros((num_reqs, rows), dtype=torch.int64, device=device)
        draft_sampled[:, 1:] = drafts
        draft_sampled = draft_sampled.view(-1)
    else:
        backing = torch.zeros(64, dtype=torch.int64, device=device)  # one 512 B block
        backing[: num_reqs * num_steps] = drafts.view(-1)
        if mode == "short_stale":
            backing[num_reqs * num_steps :] = 1 << 40
        draft_sampled = backing[: num_reqs * num_steps]
    idx = torch.arange(num_reqs, dtype=torch.int32, device=device)
    expanded = idx.repeat_interleave(rows)
    local_pos = torch.arange(rows, dtype=torch.int32, device=device).repeat(num_reqs)
    cu = torch.arange(0, num_logits + 1, rows, dtype=torch.int32, device=device)
    pos = torch.arange(num_logits, dtype=torch.int64, device=device) + 64
    temperature = torch.full((num_reqs,), 0.8, dtype=torch.float32, device=device)
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=device) * 7919 + 13
    dead_cut = torch.full((num_reqs,), num_steps, dtype=torch.int32, device=device)
    sampled, num_sampled = rejection_sample(
        target, draft_logits, draft_sampled, cu, pos, idx, expanded, local_pos,
        temperature, seeds, num_steps, None,
        use_fp64=False, use_block_verification=False, dead_row_cut=dead_cut,
    )
    torch.cuda.synchronize()
    print(json.dumps({
        "mode": mode, "draft_sampled_numel": int(draft_sampled.numel()),
        "num_logits": num_logits, "num_sampled": num_sampled.tolist(),
        "max_sampled_token": int(sampled.max().item()),
    }))


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--mode":
        run(sys.argv[2])
        return
    report = []
    for mode in ("rows", "short_zero", "short_stale"):
        p = subprocess.run(
            [sys.executable, sys.argv[0], "--mode", mode],
            capture_output=True, text=True, timeout=600,
        )
        err = [l for l in p.stderr.splitlines() if "rror" in l][-3:]
        report.append({"mode": mode, "exit": p.returncode,
                       "stdout": p.stdout.strip().splitlines()[-1:] if p.stdout.strip() else [],
                       "errors": err})
        print(json.dumps(report[-1]), flush=True)
    print("SUMMARY " + json.dumps(report))


if __name__ == "__main__":
    main()
