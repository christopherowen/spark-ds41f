import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu import model_runner as mr_mod
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

assert hasattr(mr_mod.GPUModelRunner, "_samples_split_from_shards"), (
    "Patched model_runner.py not mounted!"
)


def _make_req_states(max_reqs: int, max_model_len: int, num_spec_steps: int, vocab_size: int, device: torch.device):
    # Minimal RequestState mock matching what Sampler & RejectionSampler read
    rs = SimpleNamespace()
    rs.max_num_reqs = max_reqs
    rs.vocab_size = vocab_size
    rs.num_prompt_tokens = np.ones((max_reqs,), dtype=np.int32)
    rs.output_len = SimpleNamespace(gpu=torch.zeros((max_reqs,), dtype=torch.int32, device=device))
    rs.max_model_len = max_model_len
    rs.num_speculative_steps = num_spec_steps
    rs.device = device
    rs.prefill_len = SimpleNamespace(
        gpu=torch.ones((max_reqs,), dtype=torch.int32, device=device),
        np=np.ones((max_reqs,), dtype=np.int32),
    )
    rs.prompt_len = SimpleNamespace(gpu=torch.ones((max_reqs,), dtype=torch.int32, device=device), np=np.ones((max_reqs,), dtype=np.int32))
    rs.num_computed_prefill_tokens = np.ones((max_reqs,), dtype=np.int32)
    rs.all_token_ids = SimpleNamespace(
        gpu=torch.randint(10, 500, (max_reqs, 64), dtype=torch.int32, device=device)
    )
    rs.total_len = SimpleNamespace(
        gpu=torch.full((max_reqs,), 32, dtype=torch.int32, device=device)
    )
    rs.prompt_bin_mask = torch.zeros((max_reqs, 129280 // 32 + 1), dtype=torch.int32, device=device)
    rs.output_bin_counts = torch.zeros((max_reqs, 129280), dtype=torch.int32, device=device)
    return rs


def _build_runner(vocab_size: int, max_reqs: int, num_spec_steps: int, logits_full: torch.Tensor, tp_size: int = 1):
    device = logits_full.device
    rs = _make_req_states(max_reqs, 4096, num_spec_steps, vocab_size, device)
    mr_mod.get_tensor_model_parallel_world_size = lambda: tp_size
    sampler = Sampler(
        max_num_reqs=max_reqs,
        vocab_size=vocab_size,
        device=device,
        req_states=rs,
        logprobs_mode="raw_logprobs",
        num_speculative_tokens=num_spec_steps,
    )
    sampler._tp_lockstep_sampling = lambda: True
    spec_cfg = SimpleNamespace(
        num_verification_tokens=num_spec_steps,
        num_speculative_tokens=num_spec_steps,
        enable_adaptive_verification=False,
        rejection_sample_method="block",
    )
    rej = RejectionSampler(sampler=sampler, spec_config=spec_cfg, device=device)

    gather_calls = []

    class FakeLogitsProcessor:
        use_all_gather = True
        scale = 1.0
        soft_cap = None

        def _gather_logits(self, local_t: torch.Tensor) -> torch.Tensor:
            gather_calls.append(local_t.shape[0])
            return local_t

    class FakeModel:
        def __init__(self):
            self.lm_head = SimpleNamespace(
                shard_indices=SimpleNamespace(
                    num_org_elements=vocab_size,
                    org_vocab_start_index=0,
                )
            )
            self.logits_processor = FakeLogitsProcessor()

        def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # hidden_states encodes row indices in column 0 so we return exact slice of logits_full
            rows_idx = hidden_states[:, 0].to(torch.int64)
            return logits_full.index_select(0, rows_idx)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.compute_logits_local(hidden_states)

    runner = object.__new__(mr_mod.GPUModelRunner)
    runner.model = FakeModel()
    runner.sampler = sampler
    runner.rejection_sampler = rej
    runner.vocab_size = vocab_size
    runner.device = device
    runner.sink = None
    runner.batch_sharder = None
    runner._greedy_verification = True
    runner._sampled_verification = True
    runner._cu_num_logits_np_exact = True
    runner._dead_row_cut = None
    runner._step_draft_steps = num_spec_steps
    return runner, gather_calls


def _make_batch(
    idx_mapping_np: np.ndarray,
    counts: list[int],
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    device: torch.device,
):
    num_reqs = len(counts)
    total_rows = sum(counts)
    cu_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(counts, out=cu_np[1:])
    drafts_per_req = np.array([c - 1 for c in counts], dtype=np.int32)
    num_draft_tokens = int(drafts_per_req.sum())

    exp_idx = np.repeat(idx_mapping_np, counts).astype(np.int32)
    exp_lpos = np.concatenate([np.arange(c, dtype=np.int32) for c in counts])

    return InputBatch(
        req_ids=[f"r{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=torch.tensor(idx_mapping_np, dtype=torch.int32, device=device),
        idx_mapping_np=idx_mapping_np.astype(np.int32),
        expanded_idx_mapping=torch.tensor(exp_idx, dtype=torch.int32, device=device),
        expanded_local_pos=torch.tensor(exp_lpos, dtype=torch.int32, device=device),
        num_scheduled_tokens=np.array(counts, dtype=np.int32),
        num_tokens=total_rows,
        num_tokens_after_padding=total_rows,
        num_draft_tokens=num_draft_tokens,
        num_draft_tokens_per_req=drafts_per_req if num_draft_tokens > 0 else None,
        query_start_loc=torch.tensor(cu_np, dtype=torch.int32, device=device),
        query_start_loc_np=cu_np,
        seq_lens=torch.full((num_reqs,), 40, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 40, dtype=torch.int32),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.full((num_reqs,), 34, dtype=np.int32),
        prefill_len_np=np.ones((num_reqs,), dtype=np.int32),
        num_computed_prefill_tokens_np=np.ones((num_reqs,), dtype=np.int32),
        is_prefilling_np=np.zeros((num_reqs,), dtype=bool),
        has_prefill=False,
        input_ids=draft_tokens,
        positions=positions,
        is_padding=torch.zeros((total_rows,), dtype=torch.bool, device=device),
        logits_indices=torch.arange(total_rows, dtype=torch.int32, device=device),
        cu_num_logits=torch.tensor(cu_np, dtype=torch.int32, device=device),
        cu_num_logits_np=cu_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def test_split_guards():
    device = torch.device("cuda")
    vocab = 129280
    logits = torch.randn((12, vocab), dtype=torch.bfloat16, device=device)
    runner, _ = _build_runner(vocab, 8, 5, logits)
    # 0 = greedy, 1 = temp=0.6, 2 = top_p=0.95
    runner.sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    runner.sampler.add_request(1, 1, SamplingParams(temperature=0.6, seed=7))
    runner.sampler.add_request(2, 1, SamplingParams(temperature=0.6, top_p=0.95, seed=9))
    runner.sampler.apply_staged_writes()

    drafts = torch.randint(0, vocab, (12,), dtype=torch.int32, device=device)
    pos = torch.arange(12, dtype=torch.int64, device=device)
    # All-fast -> False
    b_fast = _make_batch(np.array([0, 1]), [6, 6], drafts, pos, device)
    assert not runner._samples_split_from_shards(b_fast, None, None)
    # All-ctrl -> False
    b_ctrl = _make_batch(np.array([2]), [6], drafts[:6], pos[:6], device)
    assert not runner._samples_split_from_shards(b_ctrl, None, None)
    # Mixed -> True
    b_mix = _make_batch(np.array([2, 0]), [6, 6], drafts, pos, device)
    assert runner._samples_split_from_shards(b_mix, None, None)
    # Device-compacted adaptive -> False
    runner._cu_num_logits_np_exact = False
    assert not runner._samples_split_from_shards(b_mix, None, None)
    runner._cu_num_logits_np_exact = True
    # Grammar active -> False
    assert not runner._samples_split_from_shards(b_mix, object(), None)


@pytest.mark.parametrize("use_drafts", [False, True])
@pytest.mark.parametrize("use_dead_cut", [False, True])
def test_mixed_split_bit_exact_parity(use_drafts, use_dead_cut):
    if use_dead_cut and not use_drafts:
        pytest.skip("dead_row_cut applies only with drafts")
    device = torch.device("cuda")
    torch.manual_seed(2026)
    vocab = 129280
    # 8 interleaved requests: 2 ctrl (idx 0: top_p=0.95, idx 5: top_k=40 + rep_penalty=1.15)
    # + 6 fast (greedy and temp=0.6)
    counts = [6, 1, 4, 6, 3, 6, 6, 2] if use_drafts else [1] * 8
    total_rows = sum(counts)
    logits = (torch.randn((total_rows, vocab), dtype=torch.float32, device=device) * 2.0).to(torch.bfloat16)
    # Give sharp realistic peaks so acceptance and top-p cutoffs are non-trivial
    for r in range(total_rows):
        logits[r, (42 + r * 7) % vocab] = 20.0
        logits[r, (137 + r * 11) % vocab] = 18.5
        logits[r, (999 + r * 13) % vocab] = 17.0

    runner, _ = _build_runner(vocab, 16, 5, logits)
    # Use non-trivial slot IDs
    slots = np.array([3, 1, 7, 2, 11, 5, 9, 4], dtype=np.int32)
    params_list = [
        SamplingParams(temperature=0.6, top_p=0.95, seed=100),                  # ctrl
        SamplingParams(temperature=0.0),                                         # fast greedy
        SamplingParams(temperature=0.6, seed=102),                               # fast temp
        SamplingParams(temperature=0.8, seed=103),                               # fast temp
        SamplingParams(temperature=0.0),                                         # fast greedy
        SamplingParams(temperature=0.7, top_k=40, repetition_penalty=1.15, seed=105),  # ctrl
        SamplingParams(temperature=0.5, seed=106),                               # fast temp
        SamplingParams(temperature=0.0),                                         # fast greedy
    ]
    for s, sp in zip(slots, params_list):
        runner.sampler.add_request(int(s), 1, sp)
    runner.sampler.apply_staged_writes()

    drafts = torch.randint(0, vocab, (total_rows,), dtype=torch.int32, device=device)
    # Make some drafts match the peak token so rejection sampling accepts multiple tokens!
    for r in range(total_rows):
        if r % 2 == 0:
            drafts[r] = (42 + r * 7) % vocab
    pos = torch.arange(total_rows, dtype=torch.int64, device=device) + 100

    batch = _make_batch(slots, counts, drafts, pos, device)
    hidden = torch.arange(total_rows, dtype=torch.float32, device=device).unsqueeze(1).expand(total_rows, 8).contiguous()
    if use_dead_cut:
        runner._dead_row_cut = torch.tensor([4, 0, 2, 5, 1, 3, 5, 1], dtype=torch.int32, device=device)

    assert runner._samples_split_from_shards(batch, None, None)
    out_split = runner._sample_split_from_shards(hidden, batch)

    # Now compare EVERY SINGLE REQUEST against running it in a pure batch of its own kind!
    cu = batch.cu_num_logits_np
    for i in range(len(slots)):
        lo, hi = int(cu[i]), int(cu[i + 1])
        sub_batch = _make_batch(
            np.array([slots[i]], dtype=np.int32),
            [counts[i]],
            drafts[lo:hi],
            pos[lo:hi],
            device,
        )
        sub_hidden = hidden[lo:hi]
        is_ctrl = bool(runner.sampler.needs_non_temperature_logits_processing[slots[i]])
        old_cut = runner._dead_row_cut
        if use_dead_cut:
            runner._dead_row_cut = old_cut[i : i + 1]
        if not is_ctrl:
            if use_drafts:
                ref = runner._verify_from_shards(sub_hidden, sub_batch)
            else:
                ref = runner._sample_from_shards(sub_hidden, sub_batch)
        else:
            sub_logits = runner.model.compute_logits(sub_hidden)
            if use_drafts:
                ref = runner.rejection_sampler(sub_logits, sub_batch, None, dead_row_cut=runner._dead_row_cut)
            else:
                ref = runner.sampler(sub_logits, sub_batch)
        runner._dead_row_cut = old_cut

        assert torch.equal(out_split.num_sampled[i : i + 1], ref.num_sampled), f"req {i} num_sampled mismatch"
        assert torch.equal(out_split.num_rejected[i : i + 1], ref.num_rejected), f"req {i} num_rejected mismatch"
        n_tok = int(ref.num_sampled[0])
        ref_toks = ref.sampled_token_ids.view(1, -1)[:, :n_tok]
        got_toks = out_split.sampled_token_ids[i : i + 1].view(1, -1)[:, :n_tok]
        assert torch.equal(got_toks, ref_toks), f"req {i} (ctrl={is_ctrl}) token mismatch: {got_toks} vs {ref_toks}"


if __name__ == "__main__":
    # Run GPU latency benchmark: split vs unsplit full-batch on 8x6 and 16x6 mixed batches!
    device = torch.device("cuda")
    vocab = 129280
    print("=== End-to-End Mixed Sampling/Verification GPU Benchmark (TP=1 GPU compute only; excludes 8x-16x RoCE all-gather savings!) ===")
    for nreqs, nctrl in [(8, 1), (8, 2), (16, 1), (16, 2)]:
        counts = [6] * nreqs
        total_rows = sum(counts)
        logits = (torch.randn((total_rows, vocab), dtype=torch.float32, device=device) * 2.0).to(torch.bfloat16)
        for r in range(total_rows):
            logits[r, 42] = 20.0
            logits[r, 137] = 18.5
            logits[r, 999] = 17.0
        runner, _ = _build_runner(vocab, nreqs + 4, 5, logits)
        slots = np.arange(nreqs, dtype=np.int32)
        for i in range(nreqs):
            sp = SamplingParams(temperature=0.6, top_p=0.95, seed=i) if i < nctrl else SamplingParams(temperature=0.6, seed=i)
            runner.sampler.add_request(i, 1, sp)
        runner.sampler.apply_staged_writes()
        drafts = torch.full((total_rows,), 42, dtype=torch.int32, device=device)
        pos = torch.arange(total_rows, dtype=torch.int64, device=device) + 100
        batch = _make_batch(slots, counts, drafts, pos, device)
        hidden = torch.arange(total_rows, dtype=torch.float32, device=device).unsqueeze(1).expand(total_rows, 8).contiguous()

        # Warmup both paths
        for _ in range(4):
            _ = runner._sample_split_from_shards(hidden, batch)
            lg = runner.model.compute_logits(hidden)
            _ = runner.rejection_sampler(lg, batch, None)
        torch.cuda.synchronize()
        iters = 32
        s1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            s1[i].record()
            lg = runner.model.compute_logits(hidden)
            _ = runner.rejection_sampler(lg, batch, None)
            e1[i].record()
        torch.cuda.synchronize()
        t_full = sorted(a.elapsed_time(b)*1000 for a, b in zip(s1, e1))[iters//2]

        s2 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e2 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            s2[i].record()
            _ = runner._sample_split_from_shards(hidden, batch)
            e2[i].record()
        torch.cuda.synchronize()
        t_split = sorted(a.elapsed_time(b)*1000 for a, b in zip(s2, e2))[iters//2]
        print(f"  {nreqs:2d}x6 ({nctrl}/{nreqs} top_p=0.95): unsplit={t_full:7.1f} us  split={t_split:7.1f} us  delta={t_split-t_full:+7.1f} us ({(t_split/t_full-1)*100:+5.1f}%)  [+ RoCE all-gather reduced {nreqs}/{nctrl}x!]")


def test_split_tp3_selective_gather_rows_only():
    device = torch.device("cuda")
    vocab = 129280
    shard_vocab = (vocab + 2) // 3
    counts = [6, 6, 6, 6]  # 4 reqs: 1 ctrl (6 rows) + 3 fast (18 rows)
    total_rows = sum(counts)
    full_logits = torch.randn((total_rows, shard_vocab * 3), dtype=torch.bfloat16, device=device)
    full_logits[:, 10] = 20.0
    local_shard = full_logits[:, :shard_vocab].contiguous()

    runner, gather_calls = _build_runner(vocab, 8, 5, local_shard, tp_size=3)
    runner.model.lm_head.shard_indices = SimpleNamespace(
        num_org_elements=shard_vocab, org_vocab_start_index=0
    )
    # Override _gather_logits to return full_logits for the requested ctrl rows and record row count
    def fake_gather(t: torch.Tensor) -> torch.Tensor:
        gather_calls.append(t.shape[0])
        return torch.cat([t, t, t], dim=-1)
    runner.model.logits_processor._gather_logits = fake_gather

    # Also patch tensor_model_parallel_all_gather in greedy_verification to simulate 3 identical ranks
    from vllm.v1.worker.gpu.spec_decode import greedy_verification as gv
    orig_ag = gv.tensor_model_parallel_all_gather
    gv.tensor_model_parallel_all_gather = lambda t, dim=-1: torch.cat([t, t, t], dim=dim)
    try:
        slots = np.array([0, 1, 2, 3], dtype=np.int32)
        runner.sampler.add_request(0, 1, SamplingParams(temperature=0.6, top_p=0.95, seed=1))
        runner.sampler.add_request(1, 1, SamplingParams(temperature=0.0))
        runner.sampler.add_request(2, 1, SamplingParams(temperature=0.6, seed=2))
        runner.sampler.add_request(3, 1, SamplingParams(temperature=0.0))
        runner.sampler.apply_staged_writes()

        drafts = torch.full((total_rows,), 10, dtype=torch.int32, device=device)
        pos = torch.arange(total_rows, dtype=torch.int64, device=device)
        batch = _make_batch(slots, counts, drafts, pos, device)
        hidden = torch.arange(total_rows, dtype=torch.float32, device=device).unsqueeze(1).expand(total_rows, 8).contiguous()

        assert runner._samples_split_from_shards(batch, None, None)
        out = runner._sample_split_from_shards(hidden, batch)
        assert gather_calls == [6], f"Expected _gather_logits only on 6 ctrl rows, got {gather_calls}"
        assert out.sampled_token_ids.shape == (4, 6)
        assert out.num_sampled.shape == (4,)
        assert out.num_rejected.shape == (4,)
    finally:
        gv.tensor_model_parallel_all_gather = orig_ag
