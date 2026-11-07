#!/usr/bin/env python3
import torch
from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
    CONTROL_WORDS,
    HEADS,
    ROW,
    SCALES,
    engram_rows_gather,
    engram_rows_gather_kernel,
    engram_rows_release_kernel,
    engram_rows_wait_kernel,
)


def main() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(20261106)
    max_m = 48
    max_lookups = max_m * HEADS
    shard_start = 100
    shard_end = 50000

    rows = torch.randint(0, 256, (max_lookups, ROW), dtype=torch.uint8, device=device)
    rows = torch.where((rows == 0x7F) | (rows == 0xFF), torch.zeros_like(rows), rows).view(
        torch.float8_e4m3fn
    )
    scales = torch.randint(1, 250, (max_lookups, SCALES), dtype=torch.uint8, device=device)
    ids = torch.randint(0, 60000, (max_m, HEADS), dtype=torch.int64, device=device)
    ids[:, 0] = -1

    shapes = [
        ("tp4", sum(1 << (c * 4) for c in range(6)), 6),
        ("tp3", sum(1 << (c * 3) for c in range(8)), 8),
        ("tp1", (1 << 24) - 1, 24),
    ]
    for name, mask, cols in shapes:
        wait_k = engram_rows_wait_kernel()
        gather_k = engram_rows_gather_kernel(max_lookups, shard_start, shard_end, mask)
        release_k = engram_rows_release_kernel()
        for m in (1, 4, 16, 48):
            live = max(1, m - 1) if m > 1 else 1
            num_tokens = torch.tensor([live], dtype=torch.int32, device=device)

            ctrl_ref = torch.zeros((CONTROL_WORDS,), dtype=torch.int32, device=device)
            ctrl_ref[0] = 7
            ctrl_ref[16] = 7
            ctrl_ref[32] = 10_000_000
            out_ref = torch.empty((m, cols * ROW), dtype=torch.bfloat16, device=device)
            wait_k(ctrl_ref)
            gather_k(ids[:m], num_tokens, rows, scales, out_ref)
            release_k(ctrl_ref)

            ctrl_step = torch.zeros((CONTROL_WORDS,), dtype=torch.int32, device=device)
            ctrl_step[0] = 7
            ctrl_step[16] = 7
            ctrl_step[32] = 10_000_000
            out_step = torch.empty((m, cols * ROW), dtype=torch.bfloat16, device=device)
            engram_rows_gather(
                ids[:m],
                num_tokens,
                rows,
                scales,
                ctrl_step,
                out_step,
                shard_start,
                shard_end,
                columns=mask,
            )
            torch.cuda.synchronize()
            assert torch.equal(out_ref.view(torch.uint16), out_step.view(torch.uint16)), (
                f"Mismatch on {name} M={m}"
            )
            assert int(ctrl_step[1].item()) == 7, f"Consumed seq not updated on {name} M={m}"
            assert int(ctrl_step[56].item()) == 0, f"Retired CTAs not reset on {name} M={m}"
    print("SANITIZE_STEP_OK")


if __name__ == "__main__":
    main()
