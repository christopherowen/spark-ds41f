# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram disk reader and its TileLang gather: the old path's bits, fresh rows on
every step, rows that arrive while the main thread holds the GIL, and failures that
name themselves (a short read, and a GPU wait that times out)."""

import ctypes
import os
import subprocess
import sys
import time

import pytest
import torch

from vllm.models.deepseek_v4_1.engram_rows import EngramDiskRows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TABLE_ROWS = 4099
HEADS = 24
WEIGHT_OFFSET, SCALE_OFFSET = 4093, 123  # unaligned: the planner aligns the reads


def _table_dir() -> str:
    # O_DIRECT needs a real filesystem (not tmpfs or overlayfs): /cache on the fleet.
    return os.environ.get("ENGRAM_ROWS_TEST_DIR", "/cache/engram-rows-test")


def _write_table(directory: str, seed: int = 0):
    os.makedirs(directory, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (TABLE_ROWS, 256), generator=gen, dtype=torch.uint8)
    codes[codes == 0x7F] = 0x7E  # E4M3 NaN codes would hide the scale cases
    codes[codes == 0xFF] = 0xFE
    scales = torch.randint(110, 140, (TABLE_ROWS, 8), generator=gen, dtype=torch.uint8)
    scales[1, 0], scales[2, 3], scales[3, 7] = 0, 255, 254  # 2^-127, NaN, 2^127
    paths = []
    for name, data, offset in (("weight.bin", codes, WEIGHT_OFFSET), ("scale.bin", scales, SCALE_OFFSET)):
        path = os.path.join(directory, name)
        with open(path, "wb") as f:
            f.write(bytes(offset))
            f.write(data.numpy().tobytes())
        paths.append(path)
    return codes, scales, paths


def _reference(codes, scales, ids, live, start, end):
    """b12x's lookup in torch: value times its E8M0 scale, rounded to BF16."""
    rows = ids.cpu().long()
    local = (rows >= start) & (rows < end)
    local[live:] = False
    safe = rows.clamp(0, TABLE_ROWS - 1)
    values = codes[safe].view(torch.float8_e4m3fn).float()
    exponent = scales[safe].long()
    scale = torch.pow(2.0, (exponent - 127).double()).float()
    scale = torch.where(exponent == 0, torch.tensor(2.0**-127), scale)
    scale = torch.where(exponent == 255, torch.tensor(float("nan")), scale)
    out = (values * scale.repeat_interleave(32, -1)).bfloat16()
    out[~local] = 0
    return out.flatten(1)


def _rows(paths, start=0, end=TABLE_ROWS, tokens=16, **kwargs):
    rows = EngramDiskRows(device="cuda", table_rows=TABLE_ROWS, shard_start=start, shard_end=end,
                          max_lookups=tokens * HEADS, **kwargs)
    rows.add_source(paths[0], WEIGHT_OFFSET, scale=False)
    rows.add_source(paths[1], SCALE_OFFSET, scale=True)
    return rows


def _step(rows, ids, live):
    stream = torch.cuda.current_stream()
    num_tokens = torch.tensor([live], dtype=torch.int32, device="cuda")
    out = torch.full((ids.shape[0], HEADS * 256), 7, dtype=torch.bfloat16, device="cuda")
    rows.stage(stream, ids, ids.numel())
    rows.gather(out, ids, num_tokens)
    return out, num_tokens


def _same(a, b):
    return torch.equal(a.view(torch.int16).cpu(), b.view(torch.int16).cpu())


@pytest.mark.parametrize("shard", [(0, TABLE_ROWS), (1000, 3000)])
def test_gather_matches_b12x_lookup_bits(shard):
    codes, scales, paths = _write_table(_table_dir())
    start, end = shard
    rows = _rows(paths, start, end)
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, TABLE_ROWS, (16, HEADS), generator=gen)
    ids[0, :3] = torch.tensor([1, 2, 3])  # the scale edge cases
    ids[5, 4] = -1  # a dead lookup
    ids = ids.cuda()
    out, num_tokens = _step(rows, ids, 13)
    torch.testing.assert_close(out.cpu(), _reference(codes, scales, ids, 13, start, end),
                               rtol=0, atol=0, equal_nan=True)
    # The old path: b12x's Triton lookup over the same mapped buffers, compact rows.
    from b12x.sequence.engram._kernels import _lookup

    old = torch.empty_like(out)
    _lookup[(16, HEADS)](rows.rows_e4m3, rows.scales.device_view, ids, num_tokens, old, 16,
                         T=16, ROWS=TABLE_ROWS, START=start, END=end, COMPACT=True,
                         RESIDENT_SCALES=False)
    assert _same(out, old)
    rows.close()


def test_every_step_gets_its_own_rows():
    codes, scales, paths = _write_table(_table_dir(), seed=3)
    rows = _rows(paths, 512, 3600)
    gen = torch.Generator().manual_seed(4)
    for step in range(40):
        tokens = int(torch.randint(1, 17, (1,), generator=gen))
        live = int(torch.randint(0, tokens + 1, (1,), generator=gen))
        ids = torch.randint(0, TABLE_ROWS, (tokens, HEADS), generator=gen).cuda()
        out, _ = _step(rows, ids, live)
        expected = _reference(codes, scales, ids, live, 512, 3600)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True,
                                   msg=f"step {step}")
    assert rows.stats()["steps"] == 40
    rows.close()


def test_rows_arrive_while_the_main_thread_holds_the_gil():
    """The GPU waits for rows the reader delivers while this thread sits in a C call
    that keeps the GIL: no Python thread could have released them."""
    codes, scales, paths = _write_table(_table_dir(), seed=5)
    rows = _rows(paths, delay_s=0.3)
    ids = torch.randint(0, TABLE_ROWS, (4, HEADS)).cuda()
    out, _ = _step(rows, ids, 4)
    done = torch.cuda.Event()
    done.record()
    libc = ctypes.PyDLL(None)  # PyDLL calls keep the GIL
    libc.usleep(1_500_000)
    assert done.query(), "the gather was still waiting when the GIL came back"
    torch.testing.assert_close(out.cpu(), _reference(codes, scales, ids, 4, 0, TABLE_ROWS),
                               rtol=0, atol=0, equal_nan=True)
    rows.close()


def test_a_short_read_is_named_on_the_next_step():
    codes, scales, paths = _write_table(_table_dir(), seed=6)
    rows = _rows(paths)
    with open(paths[0], "r+b") as f:
        f.truncate(WEIGHT_OFFSET + 100)
    ids = torch.full((2, HEADS), TABLE_ROWS - 1, dtype=torch.int64, device="cuda")
    rows.stage(torch.cuda.current_stream(), ids, ids.numel())  # no gather: it would wait
    deadline = time.monotonic() + 30
    while rows.failure() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "short Engram row read" in rows.failure()
    with pytest.raises(RuntimeError, match="short Engram row read"):
        rows.stage(torch.cuda.current_stream(), ids, ids.numel())


def _child_times_out(directory: str) -> None:
    _, _, paths = _write_table(directory, seed=7)
    rows = _rows(paths, delay_s=3.0, rows_timeout_s=0.2)
    ids = torch.zeros((2, HEADS), dtype=torch.int64, device="cuda")
    out, _ = _step(rows, ids, 2)
    try:
        torch.cuda.synchronize()
    except Exception as error:
        print(f"CUDA ERROR {type(error).__name__}")
    print(f"FAILURE {rows.failure()}", flush=True)
    os._exit(0)


def test_a_timed_out_wait_stops_the_gpu_with_its_record(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "--child-timeout", _table_dir() + "-timeout"],
        capture_output=True, text=True, timeout=120,
    )
    assert "CUDA ERROR" in result.stdout, result.stdout + result.stderr
    assert "the GPU waited" in result.stdout and "for the rows of Engram step 1" in result.stdout, result.stdout


if __name__ == "__main__" and sys.argv[1:2] == ["--child-timeout"]:
    _child_times_out(sys.argv[2])
