"""Complete five-position TP1 Markov drafting, including confidence layout."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch
from paired import classify, measure
from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4_1.tilelang import vocab
from vllm.models.deepseek_v4_1.tilelang.confidence import dspark_confidence
from vllm.v1.worker.gpu.spec_decode.dspark import speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--rounds', type=int, default=120)
a = p.parse_args()
path = Path('/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_dspark_greedy_markov.py')
spec = importlib.util.spec_from_file_location('draft_fixture', path)
fixtures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)
# The physical single-rank all-gather returns this same view.
speculator.tensor_model_parallel_all_gather = lambda pairs, dim: pairs
torch.manual_seed(20261105)
num_vocab, rank, hidden, max_rows, steps = 129280, 256, 5120, 16, 5
w2 = (torch.randn((num_vocab, rank), device='cuda') * 0.05).bfloat16()
packed = vocab.pack_bf16_weight(w2)
table = (torch.randn((num_vocab, rank), device='cuda') * 0.5).bfloat16()
confidence_weight = torch.randn(hidden + rank, device='cuda') * 0.01

def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

weight_hashes = dict(w1=digest(table), w2=digest(w2), confidence=digest(confidence_weight))
flush = torch.empty(32 << 20, device='cuda', dtype=torch.uint8)
results = []
for dtype in (torch.bfloat16, torch.float32):
    head = SimpleNamespace(weight=packed.values, weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries, weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=rank, packed_bf16_exponent_base=packed.exponent_base,
        tilelang_max_rows=max_rows, tilelang_markov_head=True)
    vocab.prepare_packed_head(head, dtype)
    base = torch.randn((max_rows * steps, num_vocab), device='cuda').to(dtype)
    states_hidden = torch.randn((max_rows * steps, hidden), device='cuda', dtype=torch.bfloat16)
    input_hashes = dict(base_logits=digest(base), hidden=digest(states_hidden))

    def confidence(h, m, out=None):
        dspark_confidence(h, m, confidence_weight, out)
        return out

    model = SimpleNamespace(supports_local_draft_argmax=lambda: False,
        vocab_parallel_greedy=False, draft_id_to_target_id=None,
        draft_vocab_shard=(0, num_vocab),
        compute_draft_logits_local=lambda h: base[:h.shape[0]],
        compute_draft_logits=lambda h: base[:h.shape[0]],
        markov_embed_table=lambda: table,
        markov_embed=lambda ids: torch.nn.functional.embedding(ids, table),
        markov_bias=lambda m: vocab.project_head(head, m),
        compute_confidence=confidence,
        model=SimpleNamespace(markov_head=SimpleNamespace(markov_w2=head)))
    model.markov_greedy_pairs_local = MethodType(
        DSparkDeepseekV4ForCausalLM.markov_greedy_pairs_local, model)
    for count in (1, 8, 16):
        old = fixtures._vocab_parallel_speculator(model, count, steps, rank, False, 'cuda')
        new = fixtures._vocab_parallel_speculator(model, count, steps, rank, False, 'cuda')
        for state in (old, new):
            state.use_confidence_head = True
            state.draft_token_confidence_probs = torch.empty((count, steps), device='cuda', dtype=torch.float32)
        assert torch.equal(old.input_buffers.input_ids, new.input_buffers.input_ids)
        h = states_hidden[:count * steps]

        def reference():
            model.vocab_parallel_greedy = False
            DSparkSpeculator._sample_sequential(old, count, h)
            return old.draft_tokens, old.draft_token_confidence_probs

        def candidate():
            model.vocab_parallel_greedy = True
            DSparkSpeculator._sample_sequential(new, count, h)
            return new.draft_tokens, new.draft_token_confidence_probs

        expected = tuple(t.clone() for t in reference())
        launches = dict(reference=reference, candidate=candidate, previous=reference, null=reference)
        measured, _ = measure(launches, flush, 20261105, expected, rounds=a.rounds)
        label = f'{dtype}-{count}'
        profiles = {}
        for role in ('reference', 'candidate'):
            launches[role]()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                launches[role]()
                torch.cuda.synchronize()
            trace = a.out / (label + '-' + role + '.trace.json')
            profile.export_chrome_trace(str(trace))
            kernels = [e['name'] for e in json.loads(trace.read_text())['traceEvents']
                       if e.get('cat') == 'kernel']
            profiles[role] = dict(kernels=kernels, aten_kernels=[n for n in kernels if 'at::native' in n])
        assert not profiles['candidate']['aten_kernels'], label
        results.append(dict(name=label, rows=count, dtype=str(dtype), steps=steps,
            measured=measured, profiles=profiles,
            gate=classify(measured['log_ratios']['candidate']),
            weight_sha256=weight_hashes, input_sha256=input_hashes,
            ids_sha256=digest(old.input_buffers.input_ids)))
        print(label, results[-1]['gate'], flush=True)
        (a.out / 'result.json').write_text(json.dumps(dict(cases=results,
            shape=dict(vocab=num_vocab, rank=rank, hidden=hidden, steps=steps, capacity=max_rows),
            scope='Actual sequential sampling method; precomputed backbone/base logits shared by both roles, complete Markov and confidence work retained.'), indent=2) + '\n')
