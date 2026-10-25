# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram disk reader and its TileLang wait and gather: the old path's bits, fresh
rows on every step, other ranks' slots (rows or hash columns) never read, tables read in staging order, rows
that arrive while the main thread holds the GIL, and failures that name themselves (a
short read, a reader failure that stops the GPU at once, and a GPU wait that times
out)."""

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


def _step(rows, ids, live):
    stream = torch.cuda.current_stream()
    num_tokens = torch.tensor([live], dtype=torch.int32, device="cuda")
    width = bin(rows.columns).count("1") * 256  # the rank's columns, side by side
    out = torch.full((ids.shape[0], width), 7, dtype=torch.bfloat16, device="cuda")
    rows.stage(stream, ids, ids.numel())
    rows.gather(out, ids, num_tokens)
    return out, num_tokens


def _same(a, b):
    return torch.equal(a.view(torch.int16).cpu(), b.view(torch.int16).cpu())


@pytest.mark.parametrize("columns", [ALL_COLUMNS, TP3_RANK1])
@pytest.mark.parametrize("shard", [(0, TABLE_ROWS), (1000, 3000)])
def test_gather_matches_b12x_lookup_bits(shard, columns):
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
    if columns != ALL_COLUMNS:
        rows.close()
        return  # b12x's lookup shards by rows only
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
    (``column_shard``), or it refuses the mask."""
    from vllm.models.deepseek_v4_1.tilelang.engram_rows import column_span

    assert column_span(ALL_COLUMNS) == (0, 1, HEADS)
    assert column_span(TP3_RANK1) == (1, 3, 8)
    assert column_span(1 << 23) == (23, 1, 1)
    for columns in (0, 0b1011, (1 << 2) | (1 << 5) | (1 << 9)):
        with pytest.raises(ValueError, match="every n-th"):
            column_span(columns)


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
