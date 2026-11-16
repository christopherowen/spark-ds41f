# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram disk reader and its TileLang wait and gather, and a resident table's
gather: the lookup reference's bits (B12X's lookup, in torch), fresh
rows on every step, other ranks' slots (rows or hash columns) never read, tables read in staging order, rows
that arrive while the main thread holds the GIL, and failures that name themselves (a
short read, a reader failure that stops the GPU at once, and a GPU wait that times
out)."""

import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.engram_rows import (
    PACKED_PAGE_BYTES,
    PACKED_ROW_BYTES,
    PACKED_ROWS_PER_PAGE,
    EngramDiskRows,
    column_compaction_map,
    host_hash,
    pack_engram_columns,
    packed_table_bytes,
)
from vllm.models.deepseek_v4_1.engram_rows.rows import ROW_BYTES, SCALE_BYTES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TABLE_ROWS = 4099
HEADS = 24
ALL_COLUMNS = (1 << HEADS) - 1
TP3_RANK1 = sum(1 << column for column in range(1, HEADS, 3))  # heads 1, 4, ..., 22
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


def _reference(codes, scales, ids, live, start, end, columns=ALL_COLUMNS):
    """b12x's lookup in torch: value times its E8M0 scale, rounded to BF16, of the
    hash columns in the mask ``columns``, side by side."""
    owned = [h for h in range(HEADS) if columns >> h & 1]
    rows = ids.cpu().long()[:, owned]
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


def _step(rows, ids, live, early=None):
    stream = torch.cuda.current_stream()
    num_tokens = torch.tensor([live], dtype=torch.int32, device="cuda")
    width = bin(rows.columns).count("1") * 256  # the rank's columns, side by side
    out = torch.full((ids.shape[0], width), 7, dtype=torch.bfloat16, device="cuda")
    rows.stage(stream, ids, ids.numel(), early)
    rows.gather(out, ids, num_tokens)
    return out, num_tokens


@pytest.mark.parametrize("columns", [ALL_COLUMNS, TP3_RANK1])
@pytest.mark.parametrize("shard", [(0, TABLE_ROWS), (1000, 3000)])
def test_gather_matches_lookup_reference_bits(shard, columns):
    codes, scales, paths = _write_table(_table_dir())
    start, end = shard
    rows = _rows(paths, start, end, columns=columns)
    # Slots of other ranks' rows are never written or read: fill them with E4M3 NaN.
    rows.weights.host.fill_(0x7F)
    rows.scales.host.fill_(0xFF)
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, TABLE_ROWS, (16, HEADS), generator=gen)
    ids[0, :3] = torch.tensor([1, 2, 3])  # the scale edge cases
    ids[5, 4] = -1  # a dead lookup
    ids = ids.cuda()
    out, num_tokens = _step(rows, ids, 13)
    torch.testing.assert_close(out.cpu(), _reference(codes, scales, ids, 13, start, end, columns),
                               rtol=0, atol=0, equal_nan=True)
    rows.close()


@pytest.mark.parametrize("live", [0, 1, 13, 16, 17, -1])
@pytest.mark.parametrize("shard", [(0, TABLE_ROWS), (1000, 3000), (3000, 5099)])
def test_resident_table_gather_matches_lookup_reference_bits(shard, live):
    """A resident table's gather: its rank's row shard (the last rank's padded past the
    table), every column, zeros past the live count and for every token when the count
    is outside [0, tokens]."""
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import engram_table_gather

    codes, scales, _ = _write_table(_table_dir(), seed=9)
    start, end = shard
    last = min(end, TABLE_ROWS)  # the table's rows this shard holds end here
    table = torch.full((end - start, 256), 0x7F, dtype=torch.uint8)
    shard_scales = torch.full((end - start, 8), 0xFF, dtype=torch.uint8)
    table[: last - start] = codes[start:last]
    shard_scales[: last - start] = scales[start:last]
    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(-1, TABLE_ROWS + 1200, (16, HEADS), generator=gen)
    ids[0, :3] = torch.tensor([1, 2, 3])  # the scale edge cases
    ids[1, :4] = torch.tensor([start - 1, start, last - 1, last])
    ids = ids.cuda()
    out = torch.full((16, HEADS * 256), 7, dtype=torch.bfloat16, device="cuda")
    num_tokens = torch.tensor([live], dtype=torch.int32, device="cuda")
    weights = table.cuda().view(torch.float8_e4m3fn)
    engram_table_gather(ids, num_tokens, weights, shard_scales.cuda(), out, start, last)
    live_rows = live if 0 <= live <= 16 else 0
    expected = _reference(codes, scales, ids, live_rows, start, last)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)


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


def test_tables_are_read_in_staging_order():
    """One worker serves every table: the second table's rows follow the first's."""
    _, _, paths = _write_table(_table_dir(), seed=8)
    first, second = _rows(paths, delay_s=0.3), _rows(paths, delay_s=0.3)
    ids = torch.randint(0, TABLE_ROWS, (4, HEADS)).cuda()
    stream = torch.cuda.current_stream()
    first.stage(stream, ids, ids.numel())
    second.stage(stream, ids, ids.numel())
    published = {}
    deadline = time.monotonic() + 30
    while len(published) < 2 and time.monotonic() < deadline:
        for name, rows in (("first", first), ("second", second)):
            if name not in published and int(rows._control[rows.native.CTRL_ROWS_SEQ]) == rows._sequence:
                published[name] = time.monotonic()
    assert set(published) == {"first", "second"}
    assert published["second"] - published["first"] > 0.2, published
    first.close()
    second.close()


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
    try:  # the stopped GPU fails the next launch or the synchronize
        _step(rows, ids, 2)
        torch.cuda.synchronize()
    except Exception as error:
        print(f"CUDA ERROR {type(error).__name__}")
    print(f"FAILURE {rows.failure()}", flush=True)
    os._exit(0)


def _child_reader_fails(directory: str) -> None:
    _, _, paths = _write_table(directory, seed=9)
    rows = _rows(paths, rows_timeout_s=60.0)
    with open(paths[0], "r+b") as f:
        f.truncate(WEIGHT_OFFSET + 100)
    ids = torch.full((2, HEADS), TABLE_ROWS - 1, dtype=torch.int64, device="cuda")
    start = time.monotonic()
    try:  # the stopped GPU fails the next launch or the synchronize
        _step(rows, ids, 2)
        torch.cuda.synchronize()
    except Exception as error:
        print(f"CUDA ERROR {type(error).__name__}")
    print(f"SECONDS {time.monotonic() - start:.2f}", flush=True)
    print(f"FAILURE {rows.failure()}", flush=True)
    os._exit(0)


def _child(flag: str, directory: str):
    # The child imports vLLM as this process does (pytest runs from the checkout).
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p or os.getcwd() for p in sys.path))
    return subprocess.run([sys.executable, __file__, flag, directory],
                          capture_output=True, text=True, timeout=120, env=env)


def test_a_reader_failure_stops_the_gpu_at_once():
    """A failed read stops the GPU when the reader reports it, not at the timeout."""
    result = _child("--child-reader-fails", _table_dir() + "-fails")
    out = result.stdout
    assert "CUDA ERROR" in out, out + result.stderr
    assert "short Engram row read" in out and "for the rows of Engram step 1" in out, out
    seconds = float(out.split("SECONDS ")[1].split()[0])
    assert seconds < 20, out


def test_a_timed_out_wait_stops_the_gpu_with_its_record(tmp_path):
    result = _child("--child-timeout", _table_dir() + "-timeout")
    assert "CUDA ERROR" in result.stdout, result.stdout + result.stderr
    assert "the GPU waited" in result.stdout and "for the rows of Engram step 1" in result.stdout, result.stdout


if __name__ == "__main__" and sys.argv[1:2] == ["--child-timeout"]:
    _child_times_out(sys.argv[2])
if __name__ == "__main__" and sys.argv[1:2] == ["--child-reader-fails"]:
    _child_reader_fails(sys.argv[2])


# --- Rows the host hashed: read while the previous step computes ------------------


def _host_hash_inputs(seed=11):
    """A random Engram geometry over the real vocabulary size (so the image sentinel
    and out-of-vocabulary ids are in play)."""
    from vllm.models.deepseek_v4_1.engram_rows.host_hash import HostEngramHash

    gen = torch.Generator().manual_seed(seed)
    vocab, layers = 129280, 2
    token_map = torch.randint(0, 1 << 20, (vocab,), generator=gen, dtype=torch.int64)
    multipliers = torch.randint(
        -(1 << 62), 1 << 62, (layers, 4), generator=gen, dtype=torch.int64
    )
    primes = torch.randint(
        1 << 20, 1 << 28, (layers, HEADS), generator=gen, dtype=torch.int64
    )
    offsets = torch.randint(
        0, 1 << 30, (layers, HEADS), generator=gen, dtype=torch.int64
    )
    host = HostEngramHash(
        token_map.numpy(), multipliers.numpy(), primes.numpy(), offsets.numpy()
    )
    return gen, host, token_map, multipliers, primes, offsets


def test_host_hash_equals_the_gpu_hash():
    """The host's hash of a step's known rows equals the GPU kernel's, bit for bit:
    requests at their first chunk and later ones (with history), image tokens,
    out-of-vocabulary ids, a missing history start, and padding."""
    from vllm.models.deepseek_v4_1.engram_rows.host_hash import IMAGE_SENTINEL
    from vllm.models.deepseek_v4_1.tilelang.engram_hash import engram_hash

    gen, host, token_map, multipliers, primes, offsets = _host_hash_inputs()
    specs = [(0, 57), (37, 81), (2, 19), (0, 3)]  # (chunk start, rows) per request
    requests, ids, history, starts = [], [], [], [0]
    for start, rows in specs:
        tokens = torch.randint(
            0, 129280, (start + rows,), generator=gen, dtype=torch.int64
        )
        tokens[torch.randint(0, start + rows, (5,), generator=gen)] = IMAGE_SENTINEL
        tokens[torch.randint(0, start + rows, (3,), generator=gen)] = (
            200_000  # out of vocabulary
        )
        chunk = tokens[start:]
        ids.append(chunk)
        lookback = [int(tokens[p]) if p >= 0 else -1 for p in range(start - 3, start)]
        history.append(lookback)
        requests.append(
            (
                starts[-1],
                rows,
                tokens.numpy().astype("int32"),
                __import__("numpy").arange(start, start + rows),
            )
        )
        starts.append(starts[-1] + rows)
    live, padded = starts[-1], starts[-1] + 9
    ids = torch.cat(ids + [torch.zeros(9, dtype=torch.int64)]).cuda()
    out = torch.empty((padded, 2, HEADS), dtype=torch.int64, device="cuda")
    engram_hash(
        ids,
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(history, dtype=torch.int64, device="cuda"),
        token_map.cuda(),
        multipliers.cuda(),
        primes.cuda(),
        offsets.cuda(),
        out,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    expected = host.step_ids(padded, requests)
    assert (expected[live:] == -1).all()
    assert torch.equal(out.cpu(), torch.from_numpy(expected))


def _early_step(rows, ids, live, early):
    stream = torch.cuda.current_stream()
    num_tokens = torch.tensor([live], dtype=torch.int32, device="cuda")
    out = torch.full(
        (ids.shape[0], HEADS * 256), 7, dtype=torch.bfloat16, device="cuda"
    )
    rows.stage(stream, ids, ids.numel(), early)
    rows.gather(out, ids, num_tokens)
    return out


@pytest.mark.parametrize("known", ["all", "some"])
def test_early_rows_give_the_gpu_ids_rows(known):
    """A step whose rows the host hashed (all, or some with the rest UNKNOWN) gathers
    the bits a step read from the GPU's IDs does."""
    from vllm.models.deepseek_v4_1.engram_rows.host_hash import UNKNOWN

    codes, scales, paths = _write_table(_table_dir(), seed=21)
    rows = _rows(paths, 700, 3900, check_early=True)
    gen = torch.Generator().manual_seed(22)
    for step in range(12):
        ids = torch.randint(0, TABLE_ROWS, (16, HEADS), generator=gen)
        early = ids.clone().numpy()
        if known == "some":
            early[torch.rand(16, generator=gen).numpy() < 0.4] = UNKNOWN
        out = _early_step(rows, ids.cuda(), 14, early)
        torch.testing.assert_close(
            out.cpu(),
            _reference(codes, scales, ids, 14, 700, 3900),
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=f"step {step}",
        )
    stats = rows.stats()
    assert (
        stats["failure"] is None and stats["steps"] == 12 and stats["early_lookups"] > 0
    )
    rows.close()


def test_early_rows_wait_for_the_previous_gather():
    """The reader writes a step's early rows only after the GPU has gathered the
    previous step from the same buffers. As in serving, the next step is staged on the
    stream after the previous step's gather, while the host runs ahead of the GPU."""
    codes, scales, paths = _write_table(_table_dir(), seed=23)
    rows = _rows(paths, tokens=4)
    stream = torch.cuda.current_stream()
    first = torch.randint(0, TABLE_ROWS, (4, HEADS)).cuda()
    second = torch.randint(0, TABLE_ROWS, (4, HEADS))
    # Device copies before the GPU is kept busy: a pageable copy waits for the stream.
    second_gpu = second.cuda()
    num_tokens = torch.tensor([4], dtype=torch.int32, device="cuda")
    out1 = torch.empty((4, HEADS * 256), dtype=torch.bfloat16, device="cuda")
    out2 = torch.empty_like(out1)
    # A whole step first, so no kernel compiles while the GPU is kept busy below.
    rows.stage(stream, first, first.numel())
    rows.gather(out1, first, num_tokens)
    torch.cuda.synchronize()
    rows.stage(stream, first, first.numel())
    deadline = time.monotonic() + 30
    while (
        int(rows._control[rows.native.CTRL_ROWS_SEQ]) != rows._sequence
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    held = rows.weights.host.clone()
    torch.cuda._sleep(int(1e9))  # the GPU is still busy before step 1's gather
    rows.gather(out1, first, num_tokens)
    rows.stage(stream, second_gpu, second.numel(), second.numpy())
    time.sleep(0.2)
    assert torch.equal(rows.weights.host, held), (
        "early rows overwrote rows the GPU had not gathered"
    )
    rows.gather(out2, second_gpu, num_tokens)
    torch.testing.assert_close(
        out1.cpu(),
        _reference(codes, scales, first, 4, 0, TABLE_ROWS),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        out2.cpu(),
        _reference(codes, scales, second, 4, 0, TABLE_ROWS),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    rows.close()


def test_a_step_the_host_hashed_is_published_before_the_gpu_reaches_it():
    """All of a step's rows known early: the reader publishes it while the GPU is still
    busy before the step's staging (its row IDs have not reached the host)."""
    _, _, paths = _write_table(_table_dir(), seed=25)
    rows = _rows(paths, tokens=8)
    stream = torch.cuda.current_stream()
    ids = torch.randint(0, TABLE_ROWS, (8, HEADS))
    ids_gpu = ids.cuda()  # before the sleep: a pageable copy waits for the stream
    torch.cuda._sleep(int(2e9))  # about a second of GPU work ahead of the staging
    rows.stage(stream, ids_gpu, ids.numel(), ids.numpy())
    deadline = time.monotonic() + 30
    while (
        int(rows._control[rows.native.CTRL_ROWS_SEQ]) != rows._sequence
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert int(rows._control[rows.native.CTRL_ROWS_SEQ]) == rows._sequence
    assert int(rows._control[rows.native.CTRL_IDS_SEQ]) != rows._sequence, (
        "the GPU reached the staging first"
    )
    torch.cuda.synchronize()
    rows.close()


def test_a_mismatched_early_id_is_named():
    """Debug check: an early ID that differs from the GPU's fails the step with both."""
    _, _, paths = _write_table(_table_dir(), seed=27)
    rows = _rows(paths, tokens=2, check_early=True)
    ids = torch.full((2, HEADS), 10, dtype=torch.int64)
    early = ids.clone().numpy()
    early[1, 5] = 11
    rows.stage(
        torch.cuda.current_stream(), ids.cuda(), ids.numel(), early
    )  # no gather: it would wait
    deadline = time.monotonic() + 30
    while rows.failure() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "lookup 29: the host hashed row 11, the GPU row 10" in rows.failure()


_TIMELINE_CHILD = """
import os, sys, torch
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import test_deepseek_v41_engram_rows as t
_, _, paths = t._write_table(t._table_dir(), seed=27)
rows = t._rows(paths, tokens=4, timeline=True, delay_s=0.05)
first = torch.randint(0, t.TABLE_ROWS, (4, t.HEADS)).cuda()
for _ in range(3):
    t._step(rows, first, 4)
torch.cuda.synchronize()
records = rows.timeline()
assert [r["sequence"] for r in records] == [1, 2, 3], records
for r in records:
    assert r["submitted"] <= r["started"] <= r["ids"] <= r["read"] <= r["published"], r
    assert r["count"] == 4 * t.HEADS and r["early"] == 0
    assert 0 < r["plan_ns"] <= r["read"] - r["ids"], r
# Step 1's kernels compile on the host while its rows are read, so its wait is short;
# the GPU waited for step 2's delayed rows, and step 3's service saw that wait.
assert records[1]["gpu_sequence"] == 1, records[1]
third = records[2]
assert third["gpu_sequence"] == 2 and third["gpu_wait_ns"] > 10_000_000, third
assert rows.timeline() == []
rows.close()
"""


def test_timeline_records_each_step_and_the_gpu_wait():
    """Debug: with the timeline on, the reader records each served step in order and the
    GPU each wait's length (the timing is compiled into the wait only with the flag set,
    so this runs in a child process)."""
    import subprocess
    import sys

    env = dict(os.environ, SPARK_DS41F_ENGRAM_TIMELINE="1")
    result = subprocess.run(
        [sys.executable, "-c", _TIMELINE_CHILD, __file__],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_the_gather_takes_every_nth_column():
    """The gather writes a rank's columns side by side: every n-th from the first
    (``column_shard``), including padded column strides when 24 % tp_size != 0."""
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import column_span, row_kernel_calls

    assert column_span(ALL_COLUMNS) == (0, 1, HEADS)
    assert column_span(TP3_RANK1) == (1, 3, 8)
    tp5_rank4 = sum(1 << column for column in range(4, HEADS, 5))
    assert column_span(tp5_rank4) == (4, 5, 4)
    assert column_span(1 << 23) == (23, 1, 1)
    for columns in (0, 0b1011, (1 << 2) | (1 << 5) | (1 << 9)):
        with pytest.raises(ValueError, match="every n-th"):
            column_span(columns)
    codes, scales, paths = _write_table(_table_dir(), seed=31)
    rows = _rows(paths, 0, TABLE_ROWS, tokens=4, columns=tp5_rank4)
    ids = torch.randint(0, TABLE_ROWS, (4, HEADS), device="cuda")
    num_tokens = torch.tensor([3], dtype=torch.int32, device="cuda")
    out = torch.zeros((4, 5 * 256), dtype=torch.bfloat16, device="cuda")
    rows.stage(torch.cuda.current_stream(), ids, ids.numel())
    rows.gather(out, ids, num_tokens)
    torch.testing.assert_close(
        out[:, : 4 * 256].cpu(),
        _reference(codes, scales, ids, 3, 0, TABLE_ROWS, tp5_rank4),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.count_nonzero(out[:, 4 * 256 :]) == 0
    assert row_kernel_calls(4 * HEADS, 0, TABLE_ROWS, tp5_rank4, out_cols=5)[0][1][-1] == 5
    rows.close()


@pytest.mark.parametrize("columns", [0, 1 << HEADS])
def test_a_malformed_column_mask_is_refused(columns):
    _, _, paths = _write_table(_table_dir(), seed=29)
    with pytest.raises(ValueError, match="24-bit mask"):
        _rows(paths, tokens=4, columns=columns)


def test_timeline_is_off_by_default():
    _, _, paths = _write_table(_table_dir(), seed=28)
    rows = _rows(paths, tokens=4)
    ids = torch.randint(0, TABLE_ROWS, (4, HEADS)).cuda()
    _step(rows, ids, 4)
    torch.cuda.synchronize()
    assert not rows.timeline_enabled and rows.timeline() == []
    rows.close()


@pytest.mark.parametrize("columns", [ALL_COLUMNS, TP3_RANK1])
@pytest.mark.parametrize("shard", [(0, TABLE_ROWS), (1000, 3000)])
def test_resident_lookup_matches_reference_bits(shard, columns):
    """TileLang resident lookup dequantizes in-shard rows, zeroes out-of-shard/dead/tail
    rows, and handles E8M0 scale edge cases (0 -> 2^-127, 255 -> NaN) identically to
    the reference and disk gather."""
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
        engram_resident_lookup,
        resident_lookup_calls,
    )

    codes, scales, _ = _write_table(_table_dir(), seed=31)
    start, end = shard
    shard_rows = end - start
    weight_shard = codes[start:end].cuda().view(torch.float8_e4m3fn)
    scale_shard = scales[start:end].cuda()

    calls = resident_lookup_calls(shard_rows, start, end, columns)
    assert len(calls) == 1

    gen = torch.Generator().manual_seed(32)
    ids = torch.randint(0, TABLE_ROWS, (16, HEADS), generator=gen)
    if start == 0:
        ids[0, :3] = torch.tensor([1, 2, 3])  # scale edge cases: 0, 255, 254
    ids[5, 4] = -1  # dead lookup
    ids[6, 7] = TABLE_ROWS + 5  # past global table_rows
    ids_gpu = ids.cuda()
    num_tokens = torch.tensor([13], dtype=torch.int32, device="cuda")
    width = bin(columns).count("1") * 256
    out = torch.full((16, width), 7, dtype=torch.bfloat16, device="cuda")
    engram_resident_lookup(
        ids_gpu, num_tokens, weight_shard, scale_shard, out, start, end, columns
    )
    expected = _reference(codes, scales, ids, 13, start, end, columns)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert torch.count_nonzero(out[13:]) == 0


def test_tilelang_engram_geometry_matches_prime_and_pcg64_contract():
    from vllm.models.deepseek_v4_1.tilelang.engram_hash import (
        Caps,
        build_geometry,
    )

    geom = build_geometry(layer_ids=(1, 14), base_table_size=19, compressed_vocab_size=32)
    assert geom.layer_ids == (1, 14)
    assert len(geom.primes) == 2 and all(len(p) == HEADS for p in geom.primes)
    assert len(set(p for row in geom.primes for p in row)) == 2 * HEADS
    caps = Caps(
        device="cuda",
        max_tokens=8,
        max_seqs=2,
        max_requests=2,
        vocab_size=32,
        layer_id=1,
        tp_size=4,
        tp_rank=1,
    )
    assert caps.device.type == "cuda" and caps.tp_rank == 1
def test_row_kernel_calls_declares_single_step_kernel():
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
        engram_rows_step_kernel,
        row_kernel_calls,
    )

    calls = row_kernel_calls(4 * HEADS, 0, TABLE_ROWS, TP3_RANK1)
    assert calls == [(engram_rows_step_kernel, (4 * HEADS, 0, TABLE_ROWS, TP3_RANK1), {})]


def test_gather_kernel_uses_streaming_cache_hints_and_matches_reference():
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
        engram_rows_gather_kernel,
        engram_rows_step_kernel,
    )

    for fn in (engram_rows_step_kernel, engram_rows_gather_kernel):
        kernel = fn(4 * HEADS, 0, TABLE_ROWS, TP3_RANK1)
        src = kernel.get_kernel_source()
        assert "ld.global.cs.s64" in src
        assert "ld.global.cs.u8" in src
        assert "ld.global.cs.u32" in src
    step_src = engram_rows_step_kernel(4 * HEADS, 0, TABLE_ROWS, TP3_RANK1).get_kernel_source()
    assert "engram_wait_rows_cta" in step_src
    assert "engram_release_rows_cta" in step_src

    codes, scales, paths = _write_table(_table_dir(), seed=41)
    rows = _rows(paths, tokens=8, columns=TP3_RANK1)
    try:
        gen = torch.Generator().manual_seed(42)
        ids = torch.randint(0, TABLE_ROWS, (8, HEADS), generator=gen).cuda()
        out, _ = _step(rows, ids, 7)
        torch.cuda.synchronize()
        expected = _reference(codes, scales, ids.cpu(), 7, 0, TABLE_ROWS, TP3_RANK1)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    finally:
        rows.close()
def _cached_pages(path: str) -> int:
    import mmap

    size = os.path.getsize(path)
    page_size = os.sysconf("SC_PAGESIZE")
    pages = (size + page_size - 1) // page_size
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    vec = (ctypes.c_ubyte * pages)()
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            res = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
            assert res == 0
        finally:
            mm.close()
    return sum(v & 1 for v in vec)


def test_fadvise_dontneed_evicts_page_cache_on_registration_and_load(monkeypatch):
    from types import SimpleNamespace
    from vllm.models.deepseek_v4_1.common.engram import _read_table_rows

    codes, _, paths = _write_table(_table_dir(), seed=33)
    weight_path = paths[0]
    # Flush dirty pages so they are clean in the VFS page cache, then warm them.
    with open(weight_path, "rb") as f:
        os.fsync(f.fileno())
        _ = f.read()
    assert _cached_pages(weight_path) > 0

    rows = _rows(paths, tokens=4)
    try:
        # Source registration drops clean VFS page-cache pages for the table range.
        assert _cached_pages(weight_path) <= 2
        ids = torch.randint(0, TABLE_ROWS, (4, HEADS)).cuda()
        out, _ = _step(rows, ids, 4)
        torch.cuda.synchronize()
        while rows.stats()["queued"]:
            time.sleep(0.001)
        assert rows.stats()["fadvise_bytes"] > 0
    finally:
        rows.close()

    # Buffered _read_table_rows drops page cache by default, and retains it when disabled.
    dest = torch.empty_like(codes)
    source = SimpleNamespace(path=weight_path, offset=WEIGHT_OFFSET)
    monkeypatch.setenv("SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE", "0")
    _read_table_rows(dest, source, 0)
    assert torch.equal(dest, codes)
    assert _cached_pages(weight_path) >= 15

    monkeypatch.setenv("SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE", "1")
    _read_table_rows(dest, source, 0)
    assert torch.equal(dest, codes)
    assert _cached_pages(weight_path) <= 2
def test_mlock_pinned_and_mapped_host_buffers():
    from vllm.v1.utils import CpuGpuBuffer

    buf = CpuGpuBuffer(256, dtype=torch.int32, device=torch.device("cuda"), pin_memory=True)
    assert buf._mlocked is True
    buf.np[:4] = [11, 22, 33, 44]
    out = buf.copy_to_gpu(4)
    assert out.tolist() == [11, 22, 33, 44]

    _, _, paths = _write_table(_table_dir(), seed=30)
    rows = _rows(paths, tokens=4)
    assert rows._ids_mlocked is True
    assert rows.weights._mlocked is True
    assert rows.scales._mlocked is True
    assert rows.control._mlocked is True
    rows.close()
    assert rows._ids_mlocked is False
    assert rows.weights._mlocked is False

def test_column_masked_early_ids_skip_d2h_ids_copy_and_match_reference():
    """When all early lookups are known (including column-masked host hashes where
    unowned columns are -1) and check_early is False, stage() skips the D2H copy into
    rows.ids while still gathering bit-identical BF16 rows."""
    from vllm.models.deepseek_v4_1.engram_rows.host_hash import UNKNOWN

    codes, scales, paths = _write_table(_table_dir(), seed=35)
    rows = _rows(paths, tokens=8, columns=TP3_RANK1, check_early=False)
    try:
        rows.ids.fill_(-42)
        gen = torch.Generator().manual_seed(36)
        ids = torch.randint(0, TABLE_ROWS, (8, HEADS), generator=gen)
        early = ids.clone().numpy()
        unowned = [c for c in range(HEADS) if not ((TP3_RANK1 >> c) & 1)]
        early[:, unowned] = -1

        out = torch.full(
            (8, 8 * 256), 7, dtype=torch.bfloat16, device="cuda"
        )
        num_tokens = torch.tensor([7], dtype=torch.int32, device="cuda")
        stream = torch.cuda.current_stream()
        rows.stage(stream, ids.cuda(), ids.numel(), early)
        rows.gather(out, ids.cuda(), num_tokens)
        torch.cuda.synchronize()

        expected = _reference(codes, scales, ids, 7, 0, TABLE_ROWS, TP3_RANK1)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        # Because early had no UNKNOWN entries and check_early=False, rows.ids was not
        # overwritten by a D2H copy.
        assert (rows.ids == -42).all()

        # When early contains an UNKNOWN entry, stage() copies the GPU IDs into rows.ids.
        early_partial = early.copy()
        early_partial[3, 1] = UNKNOWN
        rows.stage(stream, ids.cuda(), ids.numel(), early_partial)
        rows.gather(out, ids.cuda(), num_tokens)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(rows.ids[: ids.numel()], ids.view(-1))
    finally:
        rows.close()


def test_early_rows_complete_disk_reads_before_previous_gather_finishes():
    """When CTRL_CONSUMED_SEQ >= previous, read_early scatters directly into mapped
    weights/scales (early_prefetched_steps == 0). When the previous gather is still in
    flight (CTRL_CONSUMED_SEQ < previous), read_early stages into early_weights /
    early_scales while the GPU is busy, then copies into mapped buffers after consumed."""
    codes, scales, paths = _write_table(_table_dir(), seed=24)
    rows = _rows(paths, tokens=4, cache_rows=0)
    stream = torch.cuda.current_stream()
    first = torch.randint(0, TABLE_ROWS, (4, HEADS)).cuda()
    second = torch.randint(0, TABLE_ROWS, (4, HEADS))
    second_gpu = second.cuda()
    num_tokens = torch.tensor([4], dtype=torch.int32, device="cuda")
    out1 = torch.empty((4, HEADS * 256), dtype=torch.bfloat16, device="cuda")
    out2 = torch.empty_like(out1)

    # Step 1: previous step is already consumed (0 >= 0), so read_early uses the
    # zero-bounce-buffer fast path directly into mapped weights/scales.
    rows.stage(stream, first, first.numel(), first.cpu().numpy())
    rows.gather(out1, first, num_tokens)
    torch.cuda.synchronize()
    assert rows.stats()["early_prefetched_steps"] == 0

    rows.stage(stream, first, first.numel())
    deadline = time.monotonic() + 30
    while (
        int(rows._control[rows.native.CTRL_ROWS_SEQ]) != rows._sequence
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    held_weights = rows.weights.host.clone()
    held_scales = rows.scales.host.clone()
    calls_before = rows.stats()["read_calls"]

    torch.cuda._sleep(int(1.5e9))
    rows.gather(out1, first, num_tokens)
    rows.stage(stream, second_gpu, second.numel(), second.numpy())

    deadline = time.monotonic() + 30
    while (
        rows.stats()["early_prefetched_steps"] < 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)

    # Disk reads for step 3 finished into early_weights/early_scales while the GPU is
    # still sleeping before step 2's gather!
    assert rows.stats()["early_prefetched_steps"] == 1
    assert rows.stats()["read_calls"] > calls_before
    assert int(rows._control[rows.native.CTRL_CONSUMED_SEQ]) < 2
    assert torch.equal(rows.weights.host, held_weights)
    assert torch.equal(rows.scales.host, held_scales)

    rows.gather(out2, second_gpu, num_tokens)
    torch.testing.assert_close(
        out1.cpu(),
        _reference(codes, scales, first, 4, 0, TABLE_ROWS),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        out2.cpu(),
        _reference(codes, scales, second, 4, 0, TABLE_ROWS),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    rows.close()


@pytest.mark.parametrize("columns", [ALL_COLUMNS, TP3_RANK1])
def test_user_space_row_cache_hits_evicts_and_matches_reference(columns):
    """A 4-way set-associative row cache serves repeated rows from RAM with zero disk
    reads and preserves bit-exact BF16 output under set conflicts and evictions."""
    codes, scales, paths = _write_table(_table_dir(), seed=37)
    rows = _rows(paths, tokens=16, columns=columns, cache_rows=64)
    try:
        gen = torch.Generator().manual_seed(38)
        ids = torch.randint(0, TABLE_ROWS, (16, HEADS), generator=gen)
        # Force >4 distinct row IDs into the same 16-set cache bucket (set 0) to test
        # conflict handling within a single step and across steps.
        same_set = []
        for candidate in range(TABLE_ROWS):
            h = (candidate * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            if ((h ^ (h >> 32)) & 15) == 0:
                same_set.append(candidate)
                if len(same_set) == 8:
                    break
        ids[0, :8] = torch.tensor(same_set, dtype=torch.int64)
        ids_gpu = ids.cuda()

        out1, _ = _step(rows, ids_gpu, 14)
        expected = _reference(codes, scales, ids, 14, 0, TABLE_ROWS, columns)
        torch.testing.assert_close(
            out1.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )

        # Now run a step whose unique rows fit entirely within the 64-row cache.
        small_ids = torch.arange(24, dtype=torch.int64).unsqueeze(0).expand(16, -1).contiguous()
        small_gpu = small_ids.cuda()
        out_warm, _ = _step(rows, small_gpu, 16, early=small_ids.numpy())
        expected_small = _reference(codes, scales, small_ids, 16, 0, TABLE_ROWS, columns)
        torch.testing.assert_close(
            out_warm.cpu(), expected_small, rtol=0, atol=0, equal_nan=True
        )
        torch.cuda.synchronize()
        deadline = time.monotonic() + 5.0
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        stats_after_warm = rows.stats()

        # Repeating small_ids must hit 100% in the row cache with 0 disk read_calls!
        rows.weights.host.fill_(0x7F)
        rows.scales.host.fill_(0xFF)
        out_hit, _ = _step(rows, small_gpu, 16, early=small_ids.numpy())
        torch.testing.assert_close(
            out_hit.cpu(), expected_small, rtol=0, atol=0, equal_nan=True
        )
        torch.cuda.synchronize()
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        stats_after_hit = rows.stats()
        owned_per_token = bin(columns).count("1")
        assert stats_after_hit["read_calls"] == stats_after_warm["read_calls"]
        assert (
            stats_after_hit["cache_hits"] - stats_after_warm["cache_hits"]
            == 16 * owned_per_token
        )
        assert stats_after_hit["cache_misses"] == stats_after_warm["cache_misses"]
    finally:
        rows.close()


def test_background_prefetch_warms_row_cache_without_touching_step_buffers():
    """rows.prefetch(ids) populates the user-space row cache on the worker thread without
    modifying mapped weights/scales or control sequence words, so the subsequent step
    hits 100% in cache with zero disk reads."""
    codes, scales, paths = _write_table(_table_dir(), seed=43)
    rows = _rows(paths, tokens=8, cache_rows=4096)
    try:
        rows.weights.host.fill_(0x5A)
        rows.scales.host.fill_(0xA5)
        held_weights = rows.weights.host.clone()
        held_scales = rows.scales.host.clone()

        ids = torch.randint(0, TABLE_ROWS, (8, HEADS), dtype=torch.int64)
        rows.prefetch(ids.numpy())
        deadline = time.monotonic() + 10.0
        while rows.stats()["prefetch_steps"] < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        s_pref = rows.stats()
        assert s_pref["prefetch_steps"] == 1
        assert s_pref["read_calls"] > 0
        assert int(rows._control[rows.native.CTRL_ROWS_SEQ]) == 0
        assert torch.equal(rows.weights.host, held_weights)
        assert torch.equal(rows.scales.host, held_scales)

        out, _ = _step(rows, ids.cuda(), 8, early=ids.numpy())
        expected = _reference(codes, scales, ids, 8, 0, TABLE_ROWS)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize()
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        s_step = rows.stats()
        assert s_step["read_calls"] == s_pref["read_calls"]
        assert s_step["cache_hits"] >= 8 * HEADS
    finally:
        rows.close()


@pytest.mark.parametrize("offset", [0, 4093, 4096 - ROW_BYTES, 4096 - ROW_BYTES - 4])
def test_packed_single_plane_rows_split_cleanly_across_page_boundaries(offset):
    """A 15-row-per-4KiB-page (15 x 264B + 136B padding) source never splits across
    4 KiB page boundaries when offset == 0, and scatters weights [0..256) and scales
    [256..264) identically to the two-plane reader even at unaligned offsets."""
    directory = _table_dir()
    weights, scales, paths = _write_table(directory, seed=31)
    base_packed = os.path.join(directory, "packed-base.bin")
    pack_engram_columns(
        paths[0],
        WEIGHT_OFFSET,
        paths[1],
        SCALE_OFFSET,
        base_packed,
        np.array([0], dtype=np.int64),
        np.array([TABLE_ROWS], dtype=np.int64),
    )
    assert os.path.getsize(base_packed) == packed_table_bytes(TABLE_ROWS)
    if offset == 0:
        packed_path = base_packed
    else:
        packed_path = os.path.join(directory, f"packed-{offset}.bin")
        with open(base_packed, "rb") as src_f:
            payload = src_f.read()
        with open(packed_path, "wb") as f:
            f.write(b"\x5a" * offset)
            f.write(payload)

    rows = EngramDiskRows(
        device="cuda",
        table_rows=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        max_lookups=16 * HEADS,
    )
    rows.add_packed_source(packed_path, offset)
    try:
        ids = torch.randint(0, TABLE_ROWS, (16, HEADS), dtype=torch.int64)
        ids[0, :8] = torch.tensor([0, 1, 14, 15, 16, 29, TABLE_ROWS - 2, TABLE_ROWS - 1])
        ids[1, :4] = torch.tensor([-1, -99, TABLE_ROWS, TABLE_ROWS + 5])
        actual, _ = _step(rows, ids.cuda(), 13)
        expected = _reference(weights, scales, ids, 13, 0, TABLE_ROWS)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    finally:
        rows.close()


@pytest.mark.parametrize(
    "tp_size,tp_rank",
    [(3, 0), (3, 1), (3, 2), (4, 0), (4, 3)],
)
def test_pack_engram_columns_and_compacted_column_map_match_two_plane_reader(
    tp_size, tp_rank
):
    """Packing a rank's interleaved hash columns at 15 rows per 4096 B NVMe page writes
    only that rank's rows with zero 4 KiB page straddling, cuts per-step disk read calls
    compared to two-plane storage, and gathers the exact same BF16 values."""
    directory = _table_dir()
    weights, scales, paths = _write_table(directory, seed=41 + tp_size * 10 + tp_rank)
    widths = np.full(HEADS, TABLE_ROWS // HEADS, dtype=np.int64)
    widths[: TABLE_ROWS % HEADS] += 1
    offsets = np.zeros(HEADS, dtype=np.int64)
    np.cumsum(widths[:-1], out=offsets[1:])

    owned_cols = tuple(range(tp_rank, HEADS, tp_size))
    col_mask = sum(1 << c for c in owned_cols)
    col_start, col_end, col_delta, owned_starts, owned_rows = column_compaction_map(
        offsets, TABLE_ROWS, owned_cols
    )
    packed_path = os.path.join(directory, f"rank-tp{tp_size}-r{tp_rank}.packed")
    pack_engram_columns(
        paths[0],
        WEIGHT_OFFSET,
        paths[1],
        SCALE_OFFSET,
        packed_path,
        owned_starts,
        owned_rows,
    )
    assert os.path.getsize(packed_path) == packed_table_bytes(int(owned_rows.sum()))

    two_plane = _rows(paths, tokens=32, columns=col_mask, pin_scales=False)
    packed_rows = EngramDiskRows(
        device="cuda",
        table_rows=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        columns=col_mask,
        max_lookups=32 * HEADS,
        check_early=True,
    )
    packed_rows.add_packed_source(
        packed_path,
        0,
        col_start=col_start,
        col_end=col_end,
        col_delta=col_delta,
    )
    try:
        rng = np.random.default_rng(100 + tp_rank)
        ids_np = np.empty((32, HEADS), dtype=np.int64)
        for c in range(HEADS):
            s = int(offsets[c])
            e = int(offsets[c + 1]) if c + 1 < HEADS else TABLE_ROWS
            ids_np[:, c] = rng.integers(s, e, size=32, dtype=np.int64)
        ids_np[0, owned_cols[0]] = int(offsets[owned_cols[0]])
        ids_np[1, owned_cols[-1]] = (
            int(offsets[owned_cols[-1] + 1]) - 1
            if owned_cols[-1] + 1 < HEADS
            else TABLE_ROWS - 1
        )
        ids_np[2, ::3] = -1
        ids = torch.from_numpy(ids_np)
        early = ids_np.copy()
        early[20:] = host_hash.UNKNOWN

        out_two, _ = _step(two_plane, ids.cuda(), 29, early=early)
        out_packed, _ = _step(packed_rows, ids.cuda(), 29, early=early)
        expected = _reference(weights, scales, ids, 29, 0, TABLE_ROWS, col_mask)
        torch.testing.assert_close(
            out_packed.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            out_packed.cpu(), out_two.cpu(), rtol=0, atol=0, equal_nan=True
        )

        torch.cuda.synchronize()
        deadline = time.monotonic() + 5.0
        while (
            two_plane.stats()["queued"] or packed_rows.stats()["queued"]
        ) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert packed_rows.stats()["read_calls"] < two_plane.stats()["read_calls"]
    finally:
        two_plane.close()
        packed_rows.close()


@pytest.mark.parametrize("delta", [-1, -(1 << 63)])
def test_packed_column_map_refuses_negative_delta_before_io(delta):
    rows = EngramDiskRows(
        device="cuda",
        table_rows=1,
        shard_start=0,
        shard_end=1,
        max_lookups=24,
        columns=1,
    )
    start = np.zeros(24, dtype=np.int64)
    end = np.zeros(24, dtype=np.int64)
    end[0] = 1
    offsets = np.zeros(24, dtype=np.int64)
    offsets[0] = delta
    try:
        with pytest.raises(ValueError):
            rows.add_packed_source(
                "must-not-open", col_start=start, col_end=end, col_delta=offsets
            )
    finally:
        rows.close()


def test_user_space_row_cache_hits_above_64_tokens():
    """The 4-way user-space row cache remains active for steps with >64 tokens (>1536
    lookups) in direct-slot read_rows/read_early paths."""
    codes, scales, paths = _write_table(_table_dir(), seed=51)
    rows = _rows(paths, tokens=128, columns=TP3_RANK1, cache_rows=4096)
    try:
        ids = (torch.arange(128 * HEADS, dtype=torch.int64) % 96).view(128, HEADS)
        ids_gpu = ids.cuda()
        out1, _ = _step(rows, ids_gpu, 128, early=ids.numpy())
        expected = _reference(codes, scales, ids, 128, 0, TABLE_ROWS, TP3_RANK1)
        torch.testing.assert_close(out1.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize()
        deadline = time.monotonic() + 5.0
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        s1 = rows.stats()

        out2, _ = _step(rows, ids_gpu, 128, early=ids.numpy())
        torch.testing.assert_close(out2.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize()
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        s2 = rows.stats()
        assert s2["read_calls"] == s1["read_calls"]
        assert s2["cache_hits"] - s1["cache_hits"] == 128 * 8
    finally:
        rows.close()


@pytest.mark.parametrize(
    "columns,out_cols",
    [
        (sum(1 << c for c in range(0, HEADS, 4)), 6),  # TP4
        (TP3_RANK1, 8),  # TP3
        (ALL_COLUMNS, 24),  # TP1
    ],
)
def test_multi_cta_step_kernel_matches_three_launch_across_batches(columns: int, out_cols: int):
    """Multi-CTA engram_rows_step_kernel matches the 3-launch wait+gather+release path bit-for-bit
    across decode and verification batch sizes in both eager and CUDA Graph modes, and leaves
    the retirement counter at zero after every step."""
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
        engram_rows_gather,
        engram_rows_gather_kernel,
        engram_rows_release_kernel,
        engram_rows_wait_kernel,
    )

    codes, scales, paths = _write_table(_table_dir(), seed=73)
    batch_sizes = (1, 2, 4, 8, 16, 18, 24, 30, 36, 42, 48)
    rows = _rows(paths, tokens=48, columns=columns)
    try:
        gen = torch.Generator().manual_seed(79)
        ids_cpu = torch.randint(-1, TABLE_ROWS + 32, (48, HEADS), generator=gen, dtype=torch.int64)
        ids_gpu = ids_cpu.cuda()
        stream = torch.cuda.current_stream()
        wait_k = engram_rows_wait_kernel()
        gather_k = engram_rows_gather_kernel(rows.weights.device_view.shape[0], 0, TABLE_ROWS, columns)
        release_k = engram_rows_release_kernel()

        for m in batch_sizes:
            live = torch.tensor([m], dtype=torch.int32, device="cuda")
            out_step = torch.full((m, out_cols * ROW_BYTES), -1.0, dtype=torch.bfloat16, device="cuda")
            out_three = torch.full((m, out_cols * ROW_BYTES), -1.0, dtype=torch.bfloat16, device="cuda")

            rows.stage(stream, ids_gpu[:m], m * HEADS, ids_cpu[:m].numpy())
            engram_rows_gather(
                ids_gpu[:m],
                live,
                rows.weights.device_view.view(torch.float8_e4m3fn),
                rows.scales.device_view,
                rows.control.device_view,
                out_step,
                0,
                TABLE_ROWS,
                columns=columns,
            )
            torch.cuda.synchronize()
            assert int(rows.control.host[56].item()) == 0
            assert int(rows.control.host[rows.native.CTRL_CONSUMED_SEQ].item()) == int(
                rows.control.host[rows.native.CTRL_ROWS_SEQ].item()
            )

            rows.stage(stream, ids_gpu[:m], m * HEADS, ids_cpu[:m].numpy())
            wait_k(rows.control.device_view)
            gather_k(ids_gpu[:m], live, rows.weights.device_view.view(torch.float8_e4m3fn), rows.scales.device_view, out_three)
            release_k(rows.control.device_view)
            torch.cuda.synchronize()

            assert torch.equal(out_step.view(torch.uint16), out_three.view(torch.uint16))
            expected = _reference(codes, scales, ids_cpu[:m], m, 0, TABLE_ROWS, columns)
            torch.testing.assert_close(out_step.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    finally:
        rows.close()


@pytest.mark.parametrize("offset", [0, 4096 - ROW_BYTES - 4])
def test_packed_row_cache_direct_insert_and_duplicate_miss_dedup(offset: int):
    """Packed row cache inserts directly from io_uring buffers on aligned 264B rows,
    stages split rows across 4 KiB page boundaries on unaligned offsets, deduplicates
    repeated in-step misses, and serves subsequent hits with zero disk reads."""
    directory = _table_dir()
    weights, scales, paths = _write_table(directory, seed=83 + offset)
    base_packed = os.path.join(directory, f"packed-cache-base-{offset}.bin")
    pack_engram_columns(
        paths[0],
        WEIGHT_OFFSET,
        paths[1],
        SCALE_OFFSET,
        base_packed,
        np.array([0], dtype=np.int64),
        np.array([TABLE_ROWS], dtype=np.int64),
    )
    if offset == 0:
        packed_path = base_packed
    else:
        packed_path = os.path.join(directory, f"packed-cache-{offset}.bin")
        with open(base_packed, "rb") as src_f:
            payload = src_f.read()
        with open(packed_path, "wb") as f:
            f.write(b"\x5a" * offset)
            f.write(payload)

    rows = EngramDiskRows(
        device="cuda",
        table_rows=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        columns=TP3_RANK1,
        max_lookups=32 * HEADS,
        cache_rows=256,
    )
    rows.add_packed_source(packed_path, offset)
    try:
        # 24 unique row IDs repeated across all 32 tokens (heavy in-step duplicate misses).
        base_row = torch.arange(24, dtype=torch.int64) * 15 + 14  # includes page-boundary rows
        ids = base_row.unsqueeze(0).expand(32, -1).contiguous()
        ids_gpu = ids.cuda()
        expected = _reference(weights, scales, ids, 32, 0, TABLE_ROWS, TP3_RANK1)

        out_miss, _ = _step(rows, ids_gpu, 32, early=ids.numpy())
        torch.testing.assert_close(out_miss.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize()
        deadline = time.monotonic() + 5.0
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        s_after_miss = rows.stats()
        assert s_after_miss["read_calls"] > 0

        # Poison host buffers to prove the second step comes 100% from the row cache.
        rows.weights.host.fill_(0x7F)
        rows.scales.host.fill_(0xFF)
        out_hit, _ = _step(rows, ids_gpu, 32, early=ids.numpy())
        torch.testing.assert_close(out_hit.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize()
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)
        s_after_hit = rows.stats()
        assert s_after_hit["read_calls"] == s_after_miss["read_calls"]
        assert s_after_hit["cache_hits"] - s_after_miss["cache_hits"] == 32 * 8
    finally:
        rows.close()


def test_row_cache_clock_pro_prefill_shield_preserves_decode_working_set():
    """2-bit Clock + prefill probationary insertion preserves hot multi-hit decode
    rows across a multi-chunk prefill flood (>30x cache capacity) while graduating
    repeated prompt n-grams."""
    big_rows = 4096
    rng = np.random.default_rng(109)
    root = _table_dir()
    codes = rng.integers(0, 256, size=(big_rows, ROW_BYTES), dtype=np.uint8)
    codes[codes == 0x7F] = 1
    codes[codes == 0xFF] = 1
    scales = rng.integers(120, 134, size=(big_rows, SCALE_BYTES), dtype=np.uint8)
    w_path = os.path.join(root, "big_weights.bin")
    s_path = os.path.join(root, "big_scales.bin")
    with open(w_path, "wb") as f:
        f.write(codes.tobytes())
    with open(s_path, "wb") as f:
        f.write(scales.tobytes())

    rows = EngramDiskRows(
        device="cuda",
        table_rows=big_rows,
        shard_start=0,
        shard_end=big_rows,
        max_lookups=128 * HEADS,
        columns=TP3_RANK1,
        cache_rows=256,
    )
    rows.add_source(w_path, 0, scale=False)
    rows.add_source(s_path, 0, scale=True)

    def wait_idle():
        torch.cuda.synchronize()
        deadline = time.monotonic() + 5.0
        while rows.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)

    try:
        # Build a 16-token decode step (16 * 8 = 128 distinct owned rows in 0..127).
        decode_ids = torch.zeros((16, HEADS), dtype=torch.int64)
        owned_cols = [c for c in range(HEADS) if (TP3_RANK1 >> c) & 1]
        for t in range(16):
            for idx, c in enumerate(owned_cols):
                decode_ids[t, c] = t * 8 + idx
        decode_gpu = decode_ids.cuda()
        expected_decode = _reference(
            torch.from_numpy(codes), torch.from_numpy(scales),
            decode_ids, 16, 0, big_rows, TP3_RANK1,
        )

        # Warm decode working set twice so all 128 decode rows reach usage >= 2.
        out_w1, _ = _step(rows, decode_gpu, 16, early=decode_ids.numpy())
        torch.testing.assert_close(out_w1.cpu(), expected_decode, rtol=0, atol=0)
        wait_idle()
        out_w2, _ = _step(rows, decode_gpu, 16, early=decode_ids.numpy())
        torch.testing.assert_close(out_w2.cpu(), expected_decode, rtol=0, atol=0)
        wait_idle()
        s_after_warm = rows.stats()
        assert s_after_warm["cache_hits"] == 128

        # Stream 8 prefill chunks of 128 tokens (8 * 1024 = 8192 owned lookups = 32x
        # the 256-row cache capacity) using one-time row IDs in [256, 4096), plus 8
        # recurring prompt header rows in [192, 200) on token 0 of every chunk.
        cursor = 256
        for _chunk in range(8):
            pf_ids = torch.zeros((128, HEADS), dtype=torch.int64)
            for idx, c in enumerate(owned_cols):
                pf_ids[0, c] = 192 + idx
            for t in range(1, 128):
                for idx, c in enumerate(owned_cols):
                    pf_ids[t, c] = cursor
                    cursor = 256 + ((cursor - 256 + 1) % (big_rows - 256))
            pf_gpu = pf_ids.cuda()
            out_pf, _ = _step(rows, pf_gpu, 128, early=pf_ids.numpy())
            expected_pf = _reference(
                torch.from_numpy(codes), torch.from_numpy(scales),
                pf_ids, 128, 0, big_rows, TP3_RANK1,
            )
            torch.testing.assert_close(out_pf.cpu(), expected_pf, rtol=0, atol=0)
            wait_idle()

        # Verify all 128 hot decode working-set rows survived the 32x prefill flood
        # with zero disk read calls.
        s_before_decode = rows.stats()
        out_post, _ = _step(rows, decode_gpu, 16, early=decode_ids.numpy())
        torch.testing.assert_close(out_post.cpu(), expected_decode, rtol=0, atol=0)
        wait_idle()
        s_after_decode = rows.stats()
        assert s_after_decode["cache_hits"] - s_before_decode["cache_hits"] == 128
        assert s_after_decode["read_calls"] == s_before_decode["read_calls"]
    finally:
        rows.close()


def test_two_plane_pinned_scales_halves_read_calls_and_matches_reference():
    """Default-on scale plane pinning in two-plane mode matches unpinned two-plane and
    packed outputs bit-for-bit while eliminating scale.bin O_DIRECT read calls."""
    codes, scales, paths = _write_table(_table_dir(), seed=113)
    unpinned = EngramDiskRows(
        device="cuda",
        table_rows=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        max_lookups=16 * HEADS,
        columns=TP3_RANK1,
        cache_rows=1024,
        pin_scales=False,
    )
    pinned = EngramDiskRows(
        device="cuda",
        table_rows=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        max_lookups=16 * HEADS,
        columns=TP3_RANK1,
        cache_rows=1024,
    )
    assert pinned.pin_scales is True
    for r in (unpinned, pinned):
        r.add_source(paths[0], WEIGHT_OFFSET, scale=False)
        r.add_source(paths[1], SCALE_OFFSET, scale=True)

    def wait_reader(r):
        torch.cuda.synchronize()
        deadline = time.monotonic() + 5.0
        while r.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.001)

    try:
        gen = torch.Generator().manual_seed(127)
        ids = torch.randint(0, TABLE_ROWS, (16, HEADS), generator=gen, dtype=torch.int64)
        ids_gpu = ids.cuda()
        expected = _reference(codes, scales, ids, 16, 0, TABLE_ROWS, TP3_RANK1)

        out_u1, _ = _step(unpinned, ids_gpu, 16, early=ids.numpy())
        out_p1, _ = _step(pinned, ids_gpu, 16, early=ids.numpy())
        wait_reader(unpinned)
        wait_reader(pinned)
        torch.testing.assert_close(out_u1.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(out_p1.cpu(), expected, rtol=0, atol=0)
        assert torch.equal(out_u1.view(torch.uint16), out_p1.view(torch.uint16))

        su1 = unpinned.stats()
        sp1 = pinned.stats()
        assert sp1["read_calls"] > 0
        assert sp1["read_calls"] < su1["read_calls"]
        assert sp1["read_bytes"] < su1["read_bytes"]
        assert sp1["unique_blocks"] < su1["unique_blocks"]

        # Second identical step hits 100% in the row cache with 0 additional read calls.
        out_p2, _ = _step(pinned, ids_gpu, 16, early=ids.numpy())
        wait_reader(pinned)
        torch.testing.assert_close(out_p2.cpu(), expected, rtol=0, atol=0)
        sp2 = pinned.stats()
        assert sp2["read_calls"] == sp1["read_calls"]
        assert sp2["cache_hits"] - sp1["cache_hits"] == 16 * 8
    finally:
        unpinned.close()
        pinned.close()


def test_engram_disk_rows_stage_bounds_and_idempotent_close():
    import pytest

    rows = EngramDiskRows(
        device="cuda",
        table_rows=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        max_lookups=16,
        columns=TP3_RANK1,
        cache_rows=0,
    )
    stream = torch.cuda.current_stream()
    try:
        good_ids = torch.zeros(16, dtype=torch.int64, device="cuda")
        short_ids = torch.zeros(4, dtype=torch.int64, device="cuda")
        with pytest.raises(ValueError, match="exceeds max_lookups"):
            rows.stage(stream, good_ids, 17)
        with pytest.raises(ValueError, match="exceeds max_lookups"):
            rows.stage(stream, good_ids, -1)
        with pytest.raises(ValueError, match="Engram device IDs"):
            rows.stage(stream, short_ids, 8)
    finally:
        rows.close()
        rows.close()
        assert rows._closed is True
        assert rows._reader is None
