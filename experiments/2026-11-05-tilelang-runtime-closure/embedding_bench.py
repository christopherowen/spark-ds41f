"""Public TP1 embedding: native IDs versus the exact prior normalization path."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from paired import classify, measure
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.models.deepseek_v4_1.tilelang import embedding

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'vllm.models.deepseek_v4_1.tilelang.previous_embedding', here / 'control/embedding.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
torch.manual_seed(20261105)
rows, width = 129280, 5120
original = torch.randn((rows, width), dtype=torch.bfloat16, device='cuda') * 0.05
weight_sha = hashlib.sha256(original.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
flush = torch.empty(32 << 20, device='cuda', dtype=torch.uint8)
results, unchanged = [], []
for packed in (False, True):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(original.clone(), requires_grad=False)
    from types import SimpleNamespace
    layer.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=rows,
        num_org_vocab_padding=0, added_vocab_start_index=rows, added_vocab_end_index=rows)
    layer.tp_size = 1
    candidate_method = embedding.TileLangEmbeddingMethod()
    old_method = previous.TileLangEmbeddingMethod()
    if packed:
        candidate_method.process_weights_after_loading(layer)
    # Parallel/sharded callers keep their existing CUDA program exactly.
    for dtype, symbol in ((torch.int32, 'int32'), (torch.int64, 'int64')):
        if packed:
            new = embedding._packed_kernel(width, layer.packed_bf16_exponent_base, symbol)
            old = previous._packed_kernel(width, layer.packed_bf16_exponent_base, symbol)
        else:
            words = embedding.row_words(layer.weight)
            new = embedding._kernel(words, symbol, 1)
            old = previous._kernel(words, symbol, 1)
        new_source, old_source = new.get_kernel_source(), old.get_kernel_source()
        unchanged.append(dict(packed=packed, ids=str(dtype),
            new_cuda_sha256=hashlib.sha256(new_source.encode()).hexdigest(),
            old_cuda_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
            same_cuda=new_source == old_source))
        (a.out / f'parallel-{packed}-{symbol}-new.cu').write_text(new_source)
        (a.out / f'parallel-{packed}-{symbol}-old.cu').write_text(old_source)
    for dtype in (torch.int32, torch.int64):
        for count in (1, 8, 96, 4096):
            for stride in (1, 2):
                storage = torch.randint(-8, rows, (count * stride + 1,), device='cuda', dtype=dtype)
                ids = storage[1::stride]
                assert ids.shape == (count,)
                expected = torch.nn.functional.embedding(ids.clamp_min(0).long(), original)

                def reference():
                    layer.quant_method = old_method
                    return VocabParallelEmbedding.forward(layer, ids)

                def candidate():
                    layer.quant_method = candidate_method
                    return VocabParallelEmbedding.forward(layer, ids)

                label = f'{packed}-{dtype}-{count}-{stride}'
                launches = dict(reference=reference, candidate=candidate,
                                previous=reference, null=reference)
                measured, _ = measure(launches, flush, 20261105, expected)
                profiles = {}
                for role in ('reference', 'candidate'):
                    launches[role]()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        launches[role]()
                        torch.cuda.synchronize()
                    path = a.out / (label + '-' + role + '.trace.json')
                    profile.export_chrome_trace(str(path))
                    kernels = [e['name'] for e in json.loads(path.read_text())['traceEvents']
                               if e.get('cat') == 'kernel']
                    profiles[role] = dict(kernels=kernels, aten_kernels=[n for n in kernels if 'at::native' in n])
                assert not profiles['candidate']['aten_kernels'], label
                row = dict(name=label, packed=packed, ids_dtype=str(dtype), rows=count,
                           stride=stride, input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                           measured=measured, profiles=profiles,
                           gate=classify(measured['log_ratios']['candidate']))
                results.append(row)
                print(label, row['gate'], flush=True)
                (a.out / 'result.json').write_text(json.dumps(dict(weight_sha256=weight_sha,
                    shape=[rows, width], unchanged_parallel=unchanged, cases=results), indent=2) + '\n')
