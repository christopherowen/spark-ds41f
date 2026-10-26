# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED packing, native metadata and original-coordinate DSpark regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.ced import (
    CED_WINDOW,
    KEEP_ALL,
    CEDState,
    ced_decoder_start,
    gather_rows,
    plan_ced,
    replay_rows,
    scatter_rows,
)
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadata

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_boundary_is_full_resolution_source_after_encoder():
    config = SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[4] * 20 + [1] * 20,
        kv_source_layer_ids=[0, 10, 20],
    )
    assert ced_decoder_start(config) == 20
    config.kv_source_layer_ids = [0, 10]
    assert ced_decoder_start(config) is None


def test_short_continuation_does_not_replay_old_rows():
    assert plan_ced([1, 128, 17], [False] * 3, (128, 256, 512)) is None
    plan = plan_ced([1, 4096], [False, False], (128, 256, 512, 8192))
    assert plan.capacity == 256
    assert plan.num_tokens == 129


def test_limits_shrink_rows_but_keep_the_window_bound():
    capacities = (8, 128, 256, 512, 8192)
    plan = plan_ced([7, 4096], [False, False], capacities, [128, 1])
    assert (plan.capacity, plan.num_tokens) == (8, 8)
    # The decoder keeps the mode and plans of the same step without the limit.
    assert plan.max_query_len == 128
    assert plan_ced([7, 4096], [False, False], capacities).num_tokens == 135


def test_limits_compact_short_queries():
    # A 100-row chunk that keeps 36 rows compacts, though no query passes the window.
    plan = plan_ced([7, 100], [False, False], (8, 128, 256), [128, 36])
    assert (plan.capacity, plan.num_tokens, plan.max_query_len) == (128, 43, 100)
    assert plan_ced([7, 100], [False, False], (8, 128, 256), [128, 128]) is None


def test_replay_rows_anchor_at_the_prompt_end():
    # The 16,288-row chunk and the 64-row final chunk of a 16,352-row prompt: the
    # first keeps its last 64 rows, the second all of its rows; both replay from
    # 16,224, the prompt end less the window.
    assert replay_rows([16288], [0], [16352], [0], [False]) == ([16224], [64])
    assert replay_rows([64], [16288], [16352], [0], [False]) == ([16224], [128])
    # 129 final rows: the earlier chunk keeps one row, the final chunk its last 128.
    assert replay_rows([16288], [0], [16417], [0], [False]) == ([16289], [1])
    assert replay_rows([129], [16288], [16417], [0], [False]) == ([16289], [128])
    # A short first chunk after a prefix hit, cut by the floor.
    assert replay_rows([32], [16256], [16400], [16256], [False]) == ([16272], [16])
    # A prompt shorter than the window replays from its start, or from its prefix.
    assert replay_rows([100], [0], [100], [0], [False]) == ([0], [100])
    assert replay_rows([60], [40], [100], [40], [False]) == ([40], [60])
    # A resumed request's prefill length holds the outputs it recomputes.
    assert replay_rows([4096], [4096], [8200], [4096], [False]) == ([8072], [120])
    # Decode rows keep the window from the prefix start; prompt logprobs every row.
    assert replay_rows([6], [5000], [5000], [256], [False]) == ([256], [CED_WINDOW])
    assert replay_rows([4096], [0], [8192], [64], [True]) == ([64], [KEEP_ALL])


def _aligned_chunks(threshold, start, end):
    """The scheduler's aligned prefill chunks ``(start, rows)`` of a prompt computed
    from ``start``, when every step has room for a whole chunk."""
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=threshold)
    )
    chunks = []
    while start < end:
        rows = Scheduler._aligned_prefill_chunk(
            scheduler, start, min(end - start, threshold)
        )
        chunks.append((start, rows))
        start += rows
    return chunks


def _replay_windows(chunks, prompt_end, prefix=0):
    """Each kept decoder row's first visible row, ``{position: row}``, as the requests
    kernel (kept rows, replay start) and the window metadata (the SWA window from the
    replay start) place them. Each step's draft window, whose context rows are the
    kept rows, holds only kept rows."""
    windows = {}
    for start, rows in chunks:
        floors, limits = replay_rows([rows], [start], [prompt_end], [prefix], [False])
        kept = min(rows, CED_WINDOW, limits[0])
        plan = plan_ced([rows], [False], (1 << 31,), limits)
        assert (plan is None) == (kept == rows)
        end = start + rows
        replay = min(floors[0], end - kept)
        for position in range(end - kept, end):
            windows[position] = max(position + 1 - CED_WINDOW, replay)
        draft = range(max(end - CED_WINDOW, replay), end)
        assert all(row in windows for row in draft), (start, rows, prompt_end)
    return windows


@pytest.mark.parametrize("threshold", [300, 4048, 8144, 16288])
def test_anchored_replay_does_not_depend_on_chunking(threshold):
    """However the scheduler chunks a prompt, from its start or from a prefix hit, the
    decoder replays its last CED_WINDOW rows: every row from the floor on is kept and
    sees ``[max(position - 127, floor), position]``, so the drafter's first window
    after the prompt is kept too; an earlier kept row is one chunk's last and sees only
    itself."""
    window = CED_WINDOW
    boundaries = (k * threshold + r for k in (1, 2) for r in range(-2, 2 * window + 3))
    for prompt_end in sorted({*range(1, 3 * window), *boundaries}):
        hits = (threshold - 96, threshold - 32, threshold + 32, 2 * threshold - 32)
        for prefix in (0, *(h for h in hits if 0 < h <= prompt_end - window)):
            floor = max(prefix, prompt_end - window)
            windows = _replay_windows(
                _aligned_chunks(threshold, prefix, prompt_end), prompt_end, prefix
            )
            case = (threshold, prompt_end, prefix)
            replayed = {p: first for p, first in windows.items() if p >= floor}
            expected = {p: max(p + 1 - window, floor) for p in range(floor, prompt_end)}
            assert replayed == expected, case
            earlier = [p for p, first in windows.items() if p < floor and first != p]
            assert not earlier, case


@pytest.mark.parametrize("threshold", [300, 4048, 16288])
def test_final_chunks_over_the_window_replay_as_before(threshold):
    """A prompt whose final chunk has more than CED_WINDOW rows, or that is a single
    chunk, keeps its final chunk's rows and replay start: the per-chunk anchor
    already replayed it from the prompt end."""
    for prompt_end in range(1, 3 * threshold + 2 * CED_WINDOW, 7):
        chunks = _aligned_chunks(threshold, 0, prompt_end)
        start, rows = chunks[-1]
        if len(chunks) > 1 and rows <= CED_WINDOW:
            continue
        floors, limits = replay_rows([rows], [start], [prompt_end], [0], [False])
        kept = min(rows, CED_WINDOW, limits[0])
        anchored = (kept, min(floors[0], prompt_end - kept))
        before = min(rows, CED_WINDOW)
        per_chunk = (before, prompt_end - before if rows > before else 0)
        assert anchored == per_chunk, prompt_end


def test_compacted_short_chunks_run_eagerly():
    """A step compacts a chunk that fits the graphs' query bound only when the replay
    floor cuts it; that step runs eagerly."""
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = object()
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {"decode": 0, "hit": 16256}

    def batch(rows, prompt_end, prefilling=True):
        return SimpleNamespace(
            req_ids=["decode", "hit"],
            num_scheduled_tokens=np.array([6, rows]),
            num_computed_prefill_tokens_np=np.array([5000, 16256]),
            prefill_len_np=np.array([5000, prompt_end]),
            has_prefill=prefilling,
        )

    # The 32-row first chunk after the hit keeps 16 rows, or one before the floor.
    assert state.compacts_short_queries(batch(32, 16400))
    assert state.compacts_short_queries(batch(32, 20000))
    # It keeps every row when the floor is at its start.
    assert not state.compacts_short_queries(batch(32, 16384))
    # Prompt logprobs keep every row; longer chunks are eager anyway.
    state._ced_prompt_logprobs.add("hit")
    assert not state.compacts_short_queries(batch(32, 20000))
    state._ced_prompt_logprobs.clear()
    assert not state.compacts_short_queries(batch(4096, 30000))
    assert not state.compacts_short_queries(batch(6, 5000, prefilling=False))
    assert not state.compacts_short_queries(None)


def _state(lengths, seq_lens=None, full=None, floors=None):
    """CED state staged for a step of ``lengths`` rows; ``floors``: the requests'
    replay floors (``replay_rows``)."""
    lengths = np.asarray(lengths, dtype=np.int32)
    nr = len(lengths)
    starts = torch.tensor(np.r_[0, lengths.cumsum()], dtype=torch.int32, device="cuda")
    seq = torch.tensor(
        lengths if seq_lens is None else seq_lens, dtype=torch.int32, device="cuda"
    )
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    state.stage(starts, seq, lengths, full or [False] * nr, floors or [0] * nr)
    return state, starts, seq


@cuda
def test_mixed_metadata_keeps_context_bounds_and_padding():
    # A decode request and the final chunk of an 8,192-row prompt.
    state, starts, seq = _state([1, 4096], [900, 8192], floors=[0, 8064])
    # Adaptive verification may produce fewer rows than the CPU scheduling bound.
    state.stage(starts, seq, [2, 4096], [False, False], [0, 8064])
    indices = state.get_indices()
    assert indices.numel() == 130
    expected = torch.cat(
        (torch.tensor([0], device="cuda"), torch.arange(3969, 4097, device="cuda"))
    )
    torch.testing.assert_close(indices[:129], expected)
    assert torch.all(indices[129:] == -1)
    positions = torch.cat(
        (torch.tensor([899], device="cuda"), torch.arange(4096, 8192, device="cuda"))
    )
    metadata = DeepseekV41B12xMetadata(
        4097,
        2,
        4096,
        128,
        torch.ones((4, 64), dtype=torch.int32, device="cuda"),
        starts,
        torch.tensor([899, 4096, -1, -1], device="cuda"),
        torch.tensor([4097, 2], dtype=torch.int32, device="cuda"),
        positions,
        torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                torch.ones(4096, device="cuda", dtype=torch.int32),
            )
        ),
        positions + 128,
        (positions + 1).int(),
        False,
        8192,
    )
    decoder = state.decoder_metadata(metadata)
    torch.testing.assert_close(decoder.positions[:129], positions[expected])
    assert torch.all(decoder.positions[129:] == -1)
    assert torch.all(decoder.req_id_per_token[129:] == -1)
    assert torch.all(decoder.slot_mapping[129:] == -1)
    assert torch.all(decoder.cache_lengths[129:] == 0)
    assert decoder.query_start_loc.tolist() == [0, 1, 129, 129, 129]
    assert decoder.request_positions.tolist() == [899, 8064, -1, -1]
    assert decoder.live_counts.tolist() == [129, 2]
    assert decoder.swa_replay_start.tolist() == [0, 8064, 0, 0]
    assert decoder.max_seq_len == 8192
    # Global projection metadata remains the complete encoder boundary input.
    assert metadata.positions[-1].item() == 8191
    assert metadata.query_start_loc.tolist() == [0, 1, 4097]


@cuda
def test_runtime_starts_override_scheduled_upper_bound():
    state, starts, seq = _state([1, 4096])
    starts.copy_(torch.tensor([0, 1, 4001], dtype=torch.int32, device="cuda"))
    seq.copy_(torch.tensor([1, 4000], dtype=torch.int32, device="cuda"))
    state.stage(starts, seq, [1, 4096], [False, False], [0, 0])
    indices = state.get_indices()
    assert indices[:2].tolist() == [0, 3873]
    assert indices[128].item() == 4000
    assert state.counts.tolist() == [129, 2]


@cuda
def test_prompt_logprobs_zero_keeps_every_prompt_row(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    # Isolate the CED request policy from unrelated multimodal/rope bookkeeping.
    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state, starts, seq = _state([4096, 4096])
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {}
    state.disk_engram_models, state._prefill_tokens = (), {}
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["logprobs", "ordinary"],
        query_start_loc=starts,
        seq_lens=seq,
        num_scheduled_tokens=np.array([4096, 4096]),
        num_computed_prefill_tokens_np=np.array([0, 0]),
        prefill_len_np=np.array([4096, 4096]),
    )
    state.add_request(
        0,
        SimpleNamespace(
            req_id="logprobs",
            num_computed_tokens=0,
            sampling_params=SimpleNamespace(prompt_logprobs=0),
            mm_features=[],
        ),
    )
    state._stage_ced(batch)
    indices = state.get_ced_indices()
    torch.testing.assert_close(indices[:4096], torch.arange(4096, device="cuda"))
    torch.testing.assert_close(
        indices[4096:4224], torch.arange(8064, 8192, device="cuda")
    )
    state.remove_request("logprobs")
    state._stage_ced(batch)
    assert state.get_ced_indices().numel() == 256


@cuda
def test_prefix_hit_short_chunk_has_no_unwritten_decoder_history():
    # A 128-row prompt end after a prefix hit at 4096, and a decode-like request.
    state, _, _ = _state([128, 17], [4224, 9017], floors=[4096, 0])
    assert state.get_indices() is None
    assert state.replay_start.tolist() == [4096, 0, 0, 0]


@cuda
@pytest.mark.parametrize(
    "chunk, seq_len, floor, limit, first",
    [
        # A 17-row prompt end after a prefix hit at 4096: [4096 - 111, 4096)
        # precedes the hit, and the drafter's caches are not prefix-cached.
        (17, 4096 + 17, 4096, None, 4096),
        # A chunk that ends before the floor (a 12,288-row prompt): the decoder,
        # and with it the drafter's context, kept the chunk's last row only.
        (4096, 8192, 12160, 1, 8191),
        # The chunk before a 64-row final chunk keeps its last 64 rows.
        (4096, 8192, 8128, 64, 8128),
    ],
    ids=["prefix-hit-short-chunk", "chunk-before-the-floor", "chunk-at-the-floor"],
)
def test_draft_window_starts_where_the_drafter_has_context(
    chunk, seq_len, floor, limit, first
):
    """The drafter's context rows are the CED decoder's kept rows, so a draft
    window starts at CED's replay start: before it, nothing of the request's draft
    window was written in its current blocks."""
    from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata

    state, starts, seq = _state([chunk], [seq_len], floors=[floor])
    if limit is not None:
        state.stage(starts, seq, [chunk], [False], [floor], [limit])
    assert state.replay_start[0].item() == first
    drafts, page, window, width = 6, 128, 128, 136
    draft_prefix = seq_len  # the step's anchor: drafts follow its last row
    positions = torch.arange(
        draft_prefix, draft_prefix + drafts, device="cuda", dtype=torch.int64
    )
    reqs = torch.zeros(drafts, dtype=torch.int32, device="cuda")
    table = torch.arange(1, 129, dtype=torch.int32, device="cuda").repeat(4, 1)
    draft_starts = torch.tensor(
        [0, drafts, drafts, drafts, drafts], dtype=torch.int32, device="cuda"
    )
    request_positions = torch.tensor([draft_prefix, -1, -1, -1], device="cuda")
    for replay_start, start in ((state.replay_start, first), (None, None)):
        swa = torch.full((drafts, width), 7, dtype=torch.int32, device="cuda")
        lengths = torch.full((drafts,), 7, dtype=torch.int32, device="cuda")
        top = torch.full((drafts,), 7, dtype=torch.int32, device="cuda")
        visible = torch.zeros(drafts, dtype=torch.int32, device="cuda")
        tl_metadata.chunk(
            positions, reqs, table, swa, lengths, top, visible, draft_starts,
            request_positions, 0, drafts, page=page, window=window, draft=True,
            replay_start=replay_start,
        )  # fmt: skip
        if start is None:
            # Unbounded, the window reaches back a whole window from the anchor.
            start = draft_prefix - window
        length = draft_prefix + drafts - start
        assert lengths.tolist() == [length] * drafts
        logical = torch.arange(start, start + length, device="cuda")
        expected = table[0, logical // page].long() * page + logical % page
        assert torch.equal(swa[:, :length].long(), expected.expand(drafts, -1))
        assert torch.all(swa[:, length:] == -1)


@cuda
def test_chunk_before_the_floor_keeps_its_last_row():
    # A decode request, and a chunk that ends before its prompt's floor.
    state, starts, seq = _state([7, 4096], [900, 4096], floors=[0, 8064])
    state.stage(starts, seq, [7, 4096], [False, False], [0, 8064], [128, 1])
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 4102], device="cuda")
    torch.testing.assert_close(state.get_indices(), expected)
    assert state.counts.tolist() == [8, 2]
    assert state.starts[:3].tolist() == [0, 7, 8]
    # The kept row replays only itself; the decode row keeps its whole window.
    assert state.replay_start[:2].tolist() == [0, 4095]
    assert state.request_positions[:2].tolist() == [893, 4095]


@cuda
@pytest.mark.parametrize(
    "prompt_end, steps",
    [
        # Final chunks of 4,096, 129, 128, 64 and 1 rows. Each step's kept rows (as
        # step-row indices) and replay start: every chunk keeps its rows from the
        # floor on, or its last row when it ends before the floor.
        (12288, [([4095], 4095), ([4095], 8191), (list(range(3968, 4096)), 12160)]),
        (8321, [([4095], 4095), ([4095], 8191), (list(range(1, 129)), 8193)]),
        (8320, [([4095], 4095), ([4095], 8191), (list(range(128)), 8192)]),
        (
            8256,
            [([4095], 4095), (list(range(4032, 4096)), 8128), (list(range(64)), 8128)],
        ),
        (8193, [([4095], 4095), (list(range(3969, 4096)), 8065), ([0], 8065)]),
    ],
    ids=["final-4096", "final-129", "final-128", "final-64", "final-1"],
)
def test_prompt_chunks_replay_from_the_prompt_end(monkeypatch, prompt_end, steps):
    """The model state's chunks of one prompt (threshold 4,096) through the TileLang
    requests kernel: every replayed row starts at the prompt end less the window."""
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    state._ced_prompt_logprobs, state._ced_prefix_start = set(), {}
    state.disk_engram_models, state._prefill_tokens = (), {}
    state.add_request(
        0,
        SimpleNamespace(
            req_id="long", num_computed_tokens=0, sampling_params=None, mm_features=[]
        ),
    )
    staged = []
    for start, rows in _aligned_chunks(4096, 0, prompt_end):
        starts = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        seq = torch.tensor([start + rows], dtype=torch.int32, device="cuda")
        state._stage_ced(
            SimpleNamespace(
                num_reqs=1,
                req_ids=["long"],
                query_start_loc=starts,
                seq_lens=seq,
                num_scheduled_tokens=np.array([rows]),
                num_computed_prefill_tokens_np=np.array([start]),
                prefill_len_np=np.array([prompt_end]),
            )
        )
        indices = state.get_ced_indices()
        staged.append(
            (
                list(range(rows)) if indices is None else indices.tolist(),
                state.ced_state.replay_start[0].item(),
            )
        )
    assert staged == steps


@cuda
@pytest.mark.parametrize("transpose", [False, True])
def test_packing_trailing_dimensions_invalid_rows_and_scatter(transpose):
    # scatter_rows zeroes its output with TileLang's fill of whole 16-byte packs
    # (columns.zero_), as every serving shape is: rows of 3 x 4 FP32 values here.
    source = torch.arange(7 * 3 * 4, device="cuda", dtype=torch.float32).view(7, 3, 4)
    if transpose:
        source = source.transpose(1, 2)
    indices = torch.tensor([6, 1, -1, 3, 99], device="cuda")
    packed = gather_rows(source, indices)
    expected = torch.zeros((5, *source.shape[1:]), device="cuda")
    expected[[0, 1, 3]] = source[[6, 1, 3]]
    torch.testing.assert_close(packed, expected)
    restored = scatter_rows(packed, indices, 7)
    full = torch.zeros_like(source)
    full[[6, 1, 3]] = source[[6, 1, 3]]
    torch.testing.assert_close(restored, full)


@cuda
def test_scaled_row_address_above_signed_int32():
    # High row IDs must be converted BEFORE multiplying the row/page stride.
    if torch.cuda.mem_get_info()[0] < 3 * 1024**3:
        pytest.skip("Needs a 2 GiB address-span allocation")
    source = torch.empty((2**23 + 2, 256), dtype=torch.uint8, device="cuda")
    source[2**23 + 1].fill_(173)
    indices = torch.tensor([2**23 + 1, -1], device="cuda")
    packed = gather_rows(source, indices)
    assert torch.all(packed[0] == 173)
    assert torch.all(packed[1] == 0)


@cuda
def test_frozen_packing_graph_replays_new_live_rows(monkeypatch):
    from vllm.models.deepseek_v4_1 import ced

    source = torch.arange(256 * 4, dtype=torch.float32, device="cuda").view(256, 2, 2)
    indices = torch.full((128,), -1, dtype=torch.int64, device="cuda")
    indices[:3] = torch.tensor([0, 17, 255], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        packed = gather_rows(source, indices)
        restored = scatter_rows(packed, indices, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = gather_rows(source, indices)
        restored = scatter_rows(packed, indices, 256)
    # Replay cannot consult Python or compile another packing specialization.
    monkeypatch.setattr(ced.ced_kernels, "copy_rows", None)
    indices.fill_(-1)
    indices[:2] = torch.tensor([31, 127], device="cuda")
    source.add_(1000)
    graph.replay()
    expected = torch.zeros_like(source)
    expected[[31, 127]] = source[[31, 127]]
    torch.testing.assert_close(restored, expected)


@cuda
def test_dspark_compact_context_preserves_query_anchor_and_rejections(monkeypatch):
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

    device = torch.device("cuda")
    state, starts, seq = _state([1, 4096], [11, 4096])
    target_positions = torch.cat(
        (torch.tensor([10], device=device), torch.arange(4096, device=device))
    )
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([1, 4096]),
        positions=target_positions,
        query_start_loc=starts,
        idx_mapping=torch.arange(2, device=device),
    )
    buffers = InputBuffers(4, 8192, device)
    slots = torch.empty(8192, dtype=torch.int64, device=device)
    context_positions = torch.empty_like(slots)
    context_slots = torch.empty_like(slots)
    sample_indices = torch.empty(8, dtype=torch.int64, device=device)
    sample_pos = torch.empty_like(sample_indices)
    sample_reqs = torch.empty(8, dtype=torch.int32, device=device)
    temperature = torch.ones(4, device=device)
    seeds = torch.zeros(4, dtype=torch.int64, device=device)
    table = torch.arange(1, 4 * 40 + 1, dtype=torch.int32, device=device).view(4, 40)
    table[1, 31] = 0  # Evicted/null page in retained context must remain unwritable.
    prepare_dflash_inputs(
        buffers,
        slots,
        context_positions,
        context_slots,
        sample_indices,
        sample_pos,
        sample_reqs,
        temperature,
        seeds,
        batch,
        torch.tensor([1, 1], device=device),
        torch.tensor([0, 2], device=device),
        torch.tensor([77, 88, 0, 0], device=device),
        torch.zeros(4, device=device),
        temperature,
        seeds,
        table,
        128,
        0,
        1,
        1,
        999,
        2,
        2,
        4,
        8192,
        8192,
        True,
    )
    indices = state.get_indices()
    packed_positions = gather_rows(context_positions, indices)
    packed_slots = gather_rows(context_slots, indices).masked_fill(indices < 0, -1)
    assert buffers.positions[:4].tolist() == [11, 12, 4094, 4095]
    assert buffers.input_ids[:4].tolist() == [77, 999, 88, 999]
    assert sample_pos[:4].tolist() == [12, 13, 4095, 4096]
    assert packed_positions[:3].tolist() == [10, 3968, 3969]
    assert packed_positions[127:129].tolist() == [0, 0]
    assert packed_slots[0].item() == 138
    assert torch.all(packed_slots[1:] == -1)
    # Only the first request writes; skipped prefix, null pages, rejected rows,
    # and compact padding cannot alias slot zero or resurrect stale draft KV.
    pool = torch.full((1024,), -7.0, device=device)
    writable = packed_slots >= 0
    pool[packed_slots[writable]] = 5
    assert pool[0].item() == -7
    assert torch.count_nonzero(pool != -7).item() == 1

    # Exercise propose itself: this catches packing after combine (too late),
    # packing raw rather than prepared slots, and checkpoint context overwrites.
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode import speculator as base_speculator
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash

    table[1, 31] = 72
    context_pool = torch.full((25600, 2), -7.0, device=device)
    aux = torch.arange(4097 * 2, dtype=torch.float32, device=device).view(4097, 2)
    combine_rows = []

    def combine(hidden):
        combine_rows.append(hidden.shape[0])
        return hidden + 3

    def project(hidden, positions, context_slot_mapping):
        valid = context_slot_mapping >= 0
        context_pool[context_slot_mapping[valid]] = (
            hidden[valid] + positions[valid, None]
        )

    proposer = dflash.DFlashSpeculator.__new__(dflash.DFlashSpeculator)
    proposer.model = SimpleNamespace(
        combine_hidden_states=combine, precompute_and_store_context_kv=project
    )
    proposer.model_state = SimpleNamespace(
        get_ced_indices=state.get_indices, prepare_draft_attn_metadata=lambda **_: None
    )
    proposer.hidden_states = torch.zeros((8192, 2), device=device)
    proposer.context_positions = context_positions
    proposer._context_slot_mappings = context_slots[None]
    proposer._context_preparer = SimpleNamespace(
        can_run=lambda count: pytest.fail(
            "Compact prefill entered decode context graph"
        )
    )
    proposer._layer_group_idx = None
    proposer.draft_kv_cache_group_id = 0
    proposer.draft_kv_cache_group_ids = [0]
    proposer.block_tables = SimpleNamespace(
        cp_size=1,
        get_group_cp_parameters=lambda gid: (0, 1, 1),
        slot_mappings=slots[None],
        input_block_tables=[table],
        kernel_block_sizes=[128],
    )
    proposer.input_buffers = buffers
    proposer.sample_indices = sample_indices
    proposer.sample_pos = sample_pos
    proposer.sample_idx_mapping = sample_reqs
    proposer.temperature = temperature
    proposer.seeds = seeds
    proposer.parallel_drafting_token_id = 999
    proposer.num_query_per_req = proposer.num_speculative_steps = 2
    proposer.max_num_reqs = 4
    proposer.max_num_tokens = proposer.max_model_len = 8192
    proposer._is_padding = torch.ones(8192, dtype=torch.bool, device=device)
    proposer._row_index = torch.arange(8192, device=device)
    proposer.sample_from_anchor = True
    proposer.dp_size = 1
    proposer.dp_rank = 0
    proposer.pcp_manager = None
    proposer.query_cudagraph_manager = None
    proposer.draft_attn_layer_names = ["draft"]
    proposer.draft_cp_size = 1
    proposer.kv_cache_config = None
    proposer._group_causal = False
    proposer.arange = torch.arange(5, dtype=torch.int32)
    proposer.arange_np = np.arange(5, dtype=np.int32)
    proposer.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    proposer.idx_mapping = batch.idx_mapping
    proposer.attn_groups = []

    def check_draft_boundaries(**metadata):
        torch.testing.assert_close(
            metadata["query_start_loc_cpu"],
            metadata["query_start_loc_gpu"].cpu(),
            rtol=0,
            atol=0,
        )
        assert metadata["max_query_len"] == 2
        return None

    monkeypatch.setattr(base_speculator, "build_attn_metadata", check_draft_boundaries)
    proposer._prepare_eplb_forward = lambda count: None
    proposer._generate_draft = lambda *args, **kwargs: None
    proposer.draft_tokens = torch.zeros((4, 2), device=device, dtype=torch.int64)
    batch.num_tokens = 4097
    batch.seq_lens_cpu_upper_bound = seq.cpu()
    monkeypatch.setattr(
        dflash,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            SimpleNamespace(num_reqs=2, num_tokens=4, cg_mode=CUDAGraphMode.NONE),
            None,
        ),
    )
    monkeypatch.setattr(dflash, "build_slot_mappings_by_layer", lambda *args: None)
    arguments = dict(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=aux,
        aux_hidden_states=[aux],
        num_sampled=torch.tensor([1, 1], device=device),
        num_rejected=torch.tensor([0, 2], device=device),
        last_sampled=torch.tensor([77, 88, 0, 0], device=device),
        next_prefill_tokens=torch.zeros(4, device=device),
        temperature=temperature,
        seeds=seeds,
    )
    query_capacity = proposer.max_num_reqs * proposer.num_query_per_req
    slots[query_capacity:].fill_(-3137)
    proposer.propose(**arguments)
    assert combine_rows == [129]
    expected_pool = torch.full_like(context_pool, -7)
    expected_pool[138] = aux[0] + 3 + 10
    expected_pool[72 * 128 : 72 * 128 + 126] = (
        aux[3969:4095] + 3 + torch.arange(3968, 4094, device=device)[:, None]
    )
    torch.testing.assert_close(context_pool, expected_pool)
    assert buffers.positions[:4].tolist() == [11, 12, 4094, 4095]
    assert torch.all(slots[4:query_capacity] == PAD_SLOT_ID)
    assert torch.all(slots[query_capacity:] == -3137)
    proposer.propose(**arguments, context_kv_is_restored=True)
    assert combine_rows == [129]
    torch.testing.assert_close(context_pool, expected_pool)


@cuda
def test_ced_policy_staging_does_not_synchronize_gpu():
    state, starts, seq = _state([1, 4096])
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        state.stage(starts, seq, [1, 4096], [False, False], [0, 0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert state.counts.tolist() == [129, 2]


@cuda
def test_streaming_update_preserves_private_decoder_history(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state, starts, seq = _state([1], [1025])
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {"stream": 256}
    state.disk_engram_models, state._prefill_tokens = (), {}
    # The update's one new prompt token, at 1024.
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["stream"],
        query_start_loc=starts,
        seq_lens=seq,
        num_scheduled_tokens=np.array([1]),
        num_computed_prefill_tokens_np=np.array([1024]),
        prefill_len_np=np.array([1025]),
    )
    updated = SimpleNamespace(
        req_id="stream",
        num_computed_tokens=1024,
        sampling_params=None,
        mm_features=[],
    )
    state.prepare_streaming_update("stream")
    state.remove_request("stream")
    state.add_request(0, updated)
    state._stage_ced(batch)
    # The live request's decoder rows from 256 on remain: it replays from the
    # prompt end less the window.
    assert state.ced_state.replay_start[0].item() == 1025 - CED_WINDOW
    # A genuinely new incarnation must not retain that older history bound.
    state.remove_request("stream")
    state.add_request(0, updated)
    state._stage_ced(batch)
    assert state.ced_state.replay_start[0].item() == 1024


@cuda
def test_large_full_capture_decodes_small_live_prefix(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_attn", lambda *args: {})
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state, starts, seq = _state([256])
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {}
    state.lookback_token_ids = None
    state.disk_engram_models = ()
    state.device = torch.device("cuda")
    dummy_inputs = state.prepare_dummy_inputs(1, 256)
    # Replayed with a two-token prompt.
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request"],
        query_start_loc=starts,
        seq_lens=seq,
        num_scheduled_tokens=np.array([256]),
        num_computed_prefill_tokens_np=np.array([0]),
        prefill_len_np=np.array([2]),
    )
    state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None, True)
    source = torch.ones((256, 2), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = source + 1
            indices = dummy_inputs["ced_indices"]
            if indices is not None:
                result = scatter_rows(gather_rows(result, indices), indices, 256)
        starts.copy_(torch.tensor([0, 2], dtype=torch.int32, device="cuda"))
        seq.fill_(2)
        batch.num_scheduled_tokens[:] = 2
        state._stage_ced(batch)
        source.fill_(3)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result[:2], torch.full_like(result[:2], 4))


@cuda
@pytest.mark.parametrize(
    "sink_columns, pipeline", [(True, False), (False, True), (False, False)]
)
def test_warmup_declares_the_column_scatter(sink_columns, pipeline):
    """The [hc_mult] float32 scatter is declared when the model keeps the sink
    detector's columns or a pipeline stage scatters its pre-mix, and only then."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1 import ced
    from vllm.models.deepseek_v4_1.tilelang import prepare

    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    registry = JitWarmupRegistry(None)
    with registry.activate():
        state.warmup(
            hidden_size=8, hc_mult=4, pipeline=pipeline, sink_columns=sink_columns
        )
    key = tuple(ced.ced_kernels.packed_copy_rows_key("float32", (4,), True))
    declared = [
        tuple(args)
        for function, args, _ in prepare.declared_calls(registry)
        if function is ced.ced_kernels.copy_rows_kernel
    ]
    assert (key in declared) == (sink_columns or pipeline)


@cuda
def test_startup_warmup_serves_the_sink_columns_scatter(monkeypatch):
    """A compacted decoder scatters the sink detector's [tokens, hc_mult] float32
    columns back to every row; with the model keeping them, the frozen cache
    serves that scatter without a compile."""
    import importlib

    import tilelang.cache

    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1 import ced
    from vllm.models.deepseek_v4_1.tilelang import prepare

    cache_module = importlib.import_module(tilelang.cache.freeze.__module__)
    monkeypatch.setattr(cache_module, "_frozen_reason", None)
    kernel = ced.ced_kernels.copy_rows_kernel
    monkeypatch.setattr(kernel, "_kernel_cache", {})
    monkeypatch.setattr(kernel, "_call_form_cache", type(kernel._call_form_cache)())
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    registry = JitWarmupRegistry(None)
    with registry.activate():
        state.warmup(hidden_size=8, hc_mult=4, pipeline=False, sink_columns=True)
    registry.warmup()
    prepare.warm_declared(registry)
    prepare.freeze("CED sink-columns warmup test")

    lengths = [4096, 1]
    starts = torch.tensor(
        np.r_[0, np.cumsum(lengths)], dtype=torch.int32, device="cuda"
    )
    seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    state.stage(starts, seq, lengths, [False, False], [0, 0])
    indices = state.get_indices()
    columns = torch.ones((indices.numel(), 4), dtype=torch.float32, device="cuda")
    full = scatter_rows(columns, indices, sum(lengths))
    assert full[indices].eq(1).all()
    assert full.sum().item() == 4 * indices.numel()


@cuda
def test_startup_warmup_covers_live_compaction_without_jit(monkeypatch):
    import importlib

    import tilelang.cache
    from triton import knobs

    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1 import ced

    # Freeze only this test's scope; another kernel test must be able to prepare.
    cache_module = importlib.import_module(tilelang.cache.freeze.__module__)
    monkeypatch.setattr(cache_module, "_frozen_reason", None)
    # Isolate compilation state so earlier tests cannot hide a missing warmup.
    for kernel in (
        ced.ced_kernels.requests_kernel,
        ced.ced_kernels.indices_kernel,
        ced.ced_kernels.copy_rows_kernel,
    ):
        monkeypatch.setattr(kernel, "_kernel_cache", {})
        monkeypatch.setattr(kernel, "_call_form_cache", type(kernel._call_form_cache)())
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    # The packings compile with the runner's JIT warmup, as under the worker.
    registry = JitWarmupRegistry(None)
    with registry.activate():
        state.warmup(hidden_size=8, hc_mult=4, pipeline=False, sink_columns=False)
    registry.warmup()
    from vllm.models.deepseek_v4_1.tilelang import prepare

    prepare.warm_declared(registry)
    prepare.freeze("CED index warmup test")

    def forbid_compile(**kwargs):
        raise AssertionError(
            f"CED compiled after warmup: {getattr(kwargs.get('fn'), 'name', 'unknown')}"
        )

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", forbid_compile)
    for lengths in ([129], [4096, 1], [1024, 1024, 1], [1024] * 4):
        starts_cpu = np.r_[0, np.cumsum(lengths)]
        starts = torch.tensor(starts_cpu, dtype=torch.int32, device="cuda")
        seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        state.stage(starts, seq, lengths, [False] * len(lengths), [0] * len(lengths))
        indices = state.get_indices()
        expected = torch.tensor(
            [
                row
                for begin, end in zip(starts_cpu[:-1], starts_cpu[1:])
                for row in range(max(begin, end - 128), end)
            ],
            dtype=torch.int64,
            device="cuda",
        )
        torch.testing.assert_close(indices, expected)
        source = torch.ones((sum(lengths), 8), dtype=torch.bfloat16, device="cuda")
        compact = gather_rows(source, indices)
        result = scatter_rows(compact, indices, source.shape[0])
        expected_rows = torch.zeros_like(source)
        expected_rows[expected] = 1
        torch.testing.assert_close(result, expected_rows)
        # The decoder's routing mask gathers the step's padding mask.
        is_padding = torch.zeros(sum(lengths), dtype=torch.bool, device="cuda")
        assert not gather_rows(is_padding, indices).any()
