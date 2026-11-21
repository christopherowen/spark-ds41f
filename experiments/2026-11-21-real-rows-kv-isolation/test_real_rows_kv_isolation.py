from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu import model_runner as mr_mod
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class _FakeStepTiming:
    @contextmanager
    def collect(self):
        yield []


class _FakeZeroer:
    def __init__(self, kv_caches):
        self.kv_caches = kv_caches
        self.zeroed_calls = []

    def _fill_block_ids(self, block_ids, val):
        self.zeroed_calls.append(list(block_ids))
        for c in self.kv_caches:
            for bid in block_ids:
                c[bid].fill_(val)


class _HarnessRunner:
    """Minimal GPUModelRunner harness exercising the real patched methods on CUDA."""

    _REAL_CONTEXT_CHUNK = GPUModelRunner._REAL_CONTEXT_CHUNK
    _REAL_CONTEXT_ALIGN = GPUModelRunner._REAL_CONTEXT_ALIGN
    _real_context_pool_layout = GPUModelRunner._real_context_pool_layout
    _real_context_pool_block = staticmethod(GPUModelRunner._real_context_pool_block)
    _snapshot_real_context_ring = GPUModelRunner._snapshot_real_context_ring
    _prepare_real_row_ring_blocks = GPUModelRunner._prepare_real_row_ring_blocks
    _fill_real_context = GPUModelRunner._fill_real_context
    _real_dummy_rows = GPUModelRunner._real_dummy_rows
    engine_grid = GPUModelRunner.engine_grid

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.pcp_manager = None
        self.max_model_len = 32768
        self.vocab_size = 129280
        self.step_timing = _FakeStepTiming()
        # 2 groups: group 0 = paged (block_size=256), group 1 = circular ring (block_size=256)
        self.num_blocks = 256
        self.kv_cache_config = SimpleNamespace(num_blocks=self.num_blocks)
        self.block_tables = BlockTables(
            block_sizes=[256, 256],
            max_num_reqs=8,
            max_num_batched_tokens=16384,
            max_num_blocks_per_group=[128, 128],
            device=self.device,
            kernel_block_sizes=[256, 256],
            slot_mapping_enabled=[True, True],
            slot_mapping_circular=[False, True],
        )
        # Unified pool tensors [num_blocks, 256, 4] float32 per group (disjoint slices)
        self.kv_caches = [
            torch.zeros((self.num_blocks, 256, 4), dtype=torch.float32, device=self.device)
        ]
        self.kv_block_zeroer = _FakeZeroer(self.kv_caches)
        self.input_buffers = SimpleNamespace(
            input_ids=torch.zeros(16384, dtype=torch.int32, device=self.device)
        )
        self.step_records = []

    def _make_batch(self, query_lens, depths):
        num_reqs = len(query_lens)
        num_tokens = sum(query_lens)
        q_starts = np.zeros(num_reqs + 1, dtype=np.int32)
        q_starts[1:] = np.cumsum(query_lens)
        positions = []
        for q, d in zip(query_lens, depths):
            positions.extend(range(d, d + q))
        return SimpleNamespace(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            max_query_len=max(query_lens),
            num_scheduled_tokens=np.asarray(query_lens, dtype=np.int32),
            query_start_loc_np=q_starts,
            query_start_loc=torch.tensor(q_starts, dtype=torch.int32, device=self.device),
            idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            positions=torch.tensor(positions, dtype=torch.int64, device=self.device),
            input_ids=torch.zeros(num_tokens, dtype=torch.int32, device=self.device),
            is_padding=torch.ones(num_tokens, dtype=torch.bool, device=self.device),
        )

    def _simulate_kv_step(self, query_lens, depths):
        """Run _real_dummy_rows, read each request's paged & ring history, then write
        distinct per-token KV values into the computed slots."""
        batch = self._make_batch(query_lens, depths)
        bts = tuple(self.block_tables.input_block_tables)
        slots = self._real_dummy_rows(batch, bts, depths if len(set(depths)) > 1 else depths[0])
        torch.cuda.synchronize()
        flat_kv = self.kv_caches[0].view(-1, 4)

        # Snapshot what each request READS BEFORE writing its own new tokens
        reads = []
        for r, (q, d) in enumerate(zip(query_lens, depths)):
            # Paged history read: blocks 0 .. d//256 - 1
            p_blocks = bts[0][r, : d // 256].tolist()
            p_hist = self.kv_caches[0][p_blocks].clone() if p_blocks else torch.empty(0, device=self.device)
            # Ring history read: block bts[1][r, 0]
            ring_bid = int(bts[1][r, 0].item())
            ring_before = self.kv_caches[0][ring_bid].clone()
            reads.append((p_hist, ring_before, ring_bid, list(p_blocks)))

        # Write deterministic values derived from (position, token_id) into BOTH groups' slots
        for g in range(2):
            g_slots = slots[g, : batch.num_tokens]
            pos_f = batch.positions[: batch.num_tokens].float()
            tok_f = batch.input_ids[: batch.num_tokens].float()
            vals = torch.stack([pos_f, tok_f, pos_f * 0.5 + g, tok_f * 0.25 + 1.0], dim=-1)
            flat_kv[g_slots] = vals

        # Snapshot each request's ring AFTER write and its written paged tail slots
        after = []
        q_starts = batch.query_start_loc_np
        for r, (q, d) in enumerate(zip(query_lens, depths)):
            s0, s1 = int(q_starts[r]), int(q_starts[r + 1])
            p_written = flat_kv[slots[0, s0:s1]].clone()
            ring_bid = int(bts[1][r, 0].item())
            ring_after = self.kv_caches[0][ring_bid].clone()
            after.append((p_written, ring_after, slots[0, s0:s1].tolist(), slots[1, s0:s1].tolist()))
        return batch, reads, after

    def _dummy_run(
        self,
        num_tokens: int,
        profile_num_reqs: int | None = None,
        context_len: int = 0,
        single_request_prefill: bool = False,
        real_rows: bool = False,
        **kwargs,
    ):
        if not real_rows:
            return
        n_reqs = 1 if single_request_prefill else int(profile_num_reqs or 1)
        q_per = num_tokens // n_reqs
        batch, reads, after = self._simulate_kv_step([q_per] * n_reqs, [context_len] * n_reqs)
        self.step_records.append((context_len, n_reqs, reads, after))


def test_real_rows_multi_depth_isolation_reverse_and_repeat(monkeypatch):
    monkeypatch.setattr(mr_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    runner = _HarnessRunner()

    # 1. Fill history for depths [4096, 8192]
    runner._fill_real_context(8192, depths=[4096, 8192])
    assert runner._real_context_depths == {4096, 8192}
    snap_4k_id = runner._real_context_ring_snapshots[4096][1]
    snap_8k_id = runner._real_context_ring_snapshots[8192][1]
    assert snap_4k_id != snap_8k_id

    # Snapshot ring histories MUST be distinct between 4096 and 8192!
    snap_4k_before = runner.kv_caches[0][snap_4k_id].clone()
    snap_8k_before = runner.kv_caches[0][snap_8k_id].clone()
    assert not torch.equal(snap_4k_before, snap_8k_before)
    # Paged history blocks 1..32 (8192/256 = 32 blocks in group 0)
    paged_hist_before = runner.kv_caches[0][1:33].clone()

    # 2. Individual runs: A at depth 4096 (q=6), B at depth 8192 (q=6)
    _, reads_a_solo, after_a_solo = runner._simulate_kv_step([6], [4096])
    _, reads_b_solo, after_b_solo = runner._simulate_kv_step([6], [8192])

    # 3. Batched [A(4096), B(8192)] together
    _, reads_ab, after_ab = runner._simulate_kv_step([6, 6], [4096, 8192])
    # Disjoint writable ring blocks and disjoint written slots
    assert reads_ab[0][2] != reads_ab[1][2]
    assert set(after_ab[0][2]).isdisjoint(set(after_ab[1][2]))
    assert set(after_ab[0][3]).isdisjoint(set(after_ab[1][3]))
    # Bit-exact match against individual runs
    assert torch.equal(reads_ab[0][0], reads_a_solo[0][0])
    assert torch.equal(reads_ab[0][1], reads_a_solo[0][1])
    assert torch.equal(after_ab[0][0], after_a_solo[0][0])
    assert torch.equal(after_ab[0][1], after_a_solo[0][1])
    assert torch.equal(reads_ab[1][0], reads_b_solo[0][0])
    assert torch.equal(reads_ab[1][1], reads_b_solo[0][1])
    assert torch.equal(after_ab[1][0], after_b_solo[0][0])
    assert torch.equal(after_ab[1][1], after_b_solo[0][1])

    # 4. Reversed order [B(8192), A(4096)]
    _, reads_ba, after_ba = runner._simulate_kv_step([6, 6], [8192, 4096])
    assert reads_ba[0][2] != reads_ba[1][2]
    assert torch.equal(reads_ba[0][0], reads_b_solo[0][0])
    assert torch.equal(reads_ba[0][1], reads_b_solo[0][1])
    assert torch.equal(after_ba[0][0], after_b_solo[0][0])
    assert torch.equal(after_ba[0][1], after_b_solo[0][1])
    assert torch.equal(reads_ba[1][0], reads_a_solo[0][0])
    assert torch.equal(reads_ba[1][1], reads_a_solo[0][1])
    assert torch.equal(after_ba[1][0], after_a_solo[0][0])
    assert torch.equal(after_ba[1][1], after_a_solo[0][1])

    # 5. Repeat via engine_grid with fill_real_context=False (must preserve snapshots & paged history!)
    runner.step_records.clear()
    spec = {
        "fill_real_context": False,
        "replays": 2,
        "warmup": 1,
        "settle": 1,
        "shapes": [
            {"num_tokens": 24, "num_reqs": 4, "context_len": 4096, "real_rows": True},
            {"num_tokens": 24, "num_reqs": 4, "context_len": 8192, "real_rows": True},
        ],
    }
    runner.engine_grid(spec)
    # Zeroer MUST NOT have zeroed the saved snapshot blocks!
    last_zeroed = set(runner.kv_block_zeroer.zeroed_calls[-1])
    assert snap_4k_id not in last_zeroed
    assert snap_8k_id not in last_zeroed
    # Every replay saw bit-identical paged & ring history for all 4 concurrent requests!
    assert len(runner.step_records) > 0
    for ctx, n_reqs, reads, after in runner.step_records:
        expected_ring = snap_4k_before if ctx == 4096 else snap_8k_before
        expected_paged = reads_a_solo[0][0] if ctx == 4096 else reads_b_solo[0][0]
        ring_bids = {r[2] for r in reads}
        assert len(ring_bids) == n_reqs
        assert snap_4k_id not in ring_bids and snap_8k_id not in ring_bids
        for r in reads:
            assert torch.equal(r[0], expected_paged)
            assert torch.equal(r[1], expected_ring)

    # Immutable history and snapshots remain 100% unmodified!
    assert torch.equal(runner.kv_caches[0][snap_4k_id], snap_4k_before)
    assert torch.equal(runner.kv_caches[0][snap_8k_id], snap_8k_before)
    assert torch.equal(runner.kv_caches[0][1:33], paged_hist_before)


def test_fill_real_context_false_rejects_uninitialized_depth():
    runner = _HarnessRunner()
    runner._fill_real_context(4096, depths=[4096])
    with pytest.raises(ValueError, match="fill_real_context=False requires prior"):
        runner.engine_grid(
            {
                "fill_real_context": False,
                "replays": 1,
                "warmup": 0,
                "settle": 0,
                "shapes": [
                    {"num_tokens": 6, "num_reqs": 1, "context_len": 8192, "real_rows": True}
                ],
            }
        )


def test_real_rows_virtual_block_splitting_bpk4(monkeypatch):
    monkeypatch.setattr(mr_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    runner = _HarnessRunner()
    runner.block_tables = BlockTables(
        block_sizes=[256, 256],
        max_num_reqs=8,
        max_num_batched_tokens=16384,
        max_num_blocks_per_group=[512, 128],
        device=runner.device,
        kernel_block_sizes=[64, 256],
        slot_mapping_enabled=[True, True],
        slot_mapping_circular=[False, True],
    )
    # Unified pool with 256 scheduler blocks -> 1024 64-tok kernel blocks in group 0!
    runner.kv_caches = [
        torch.zeros((runner.num_blocks * 4, 64, 4), dtype=torch.float32, device=runner.device),
        torch.zeros((runner.num_blocks, 256, 4), dtype=torch.float32, device=runner.device),
    ]
    runner.kv_block_zeroer = _FakeZeroer([runner.kv_caches[1]])

    # Fill depths [0, 4096, 8192]
    runner._fill_real_context(8192, depths=[0, 4096, 8192])
    assert runner._real_context_depths == {0, 4096, 8192}
    # Check batched 4-request step at depth 4096: all 4 paged tails and all 4 rings are strictly disjoint
    batch = runner._make_batch([6, 6, 6, 6], [4096, 4096, 4096, 4096])
    bts = tuple(runner.block_tables.input_block_tables)
    slots = runner._real_dummy_rows(batch, bts, 4096)
    torch.cuda.synchronize()
    # 4096 / 64 = 64 shared history kernel blocks (scheduler blocks 1..16 -> kernel blocks 4..67)
    hist_blocks = set(bts[0][0, :64].tolist())
    assert len(hist_blocks) == 64
    for r in range(1, 4):
        assert torch.equal(bts[0][r, :64], bts[0][0, :64])
    # Tail writable kernel block (col 64) must be distinct across all 4 reqs AND disjoint from history
    tail_blocks = [int(bts[0][r, 64].item()) for r in range(4)]
    assert len(set(tail_blocks)) == 4
    assert set(tail_blocks).isdisjoint(hist_blocks)
    # Ring blocks (col 0 of group 1) must be distinct across all 4 reqs AND disjoint from all 3 snapshots
    ring_blocks = [int(bts[1][r, 0].item()) for r in range(4)]
    snap_blocks = {runner._real_context_ring_snapshots[d][1] for d in (0, 4096, 8192)}
    assert len(set(ring_blocks)) == 4
    assert set(ring_blocks).isdisjoint(snap_blocks)
    # All 24 written slots per group must be strictly distinct
    assert len(set(slots[0, :24].tolist())) == 24
    assert len(set(slots[1, :24].tolist())) == 24
