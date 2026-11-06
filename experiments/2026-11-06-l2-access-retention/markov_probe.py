#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cache-only screen of candidate-38's native packed Markov/greedy kernel.

This is not a model step, collective, serving benchmark, or quality evaluation.
Real checkpoint weights are packed by the unchanged native packer. The fixed
inputs isolate five repeated reads; the stress case adds a 96 MiB read between
positions and does not claim that serving performs that particular sequence.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path

import torch
import tilelang.language as T
from cuda.bindings import driver as cu
from safetensors import safe_open

import vllm
from vllm.models.deepseek_v4_1.tilelang.vocab import (
    pack_bf16_weight, packed_vocab_markov_greedy_partials,
    finish_vocab_greedy_pairs, tile_config,
)
from vllm.models.glm5next.nvidia.tilelang.l2_prefetch import l2_prefetch_kernel


def check(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(result))
    return result[1] if len(result) == 2 else result[1:]


def window(stream, tensor=None, fraction=1.0, num_bytes=None, hit_prop=None):
    value = cu.CUstreamAttrValue()
    w = value.accessPolicyWindow
    w.base_ptr = 0 if tensor is None else tensor.data_ptr()
    w.num_bytes = 0 if tensor is None else (tensor.numel() * tensor.element_size() if num_bytes is None else int(num_bytes))
    w.hitRatio = fraction
    w.hitProp = cu.CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING if hit_prop is None else hit_prop
    w.missProp = cu.CUaccessProperty.CU_ACCESS_PROPERTY_NORMAL
    check(cu.cuStreamSetAttribute(
        cu.CUstream(stream.cuda_stream),
        cu.CUstreamAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, value))


def graph_kernels(graph):
    handle = cu.CUgraph(graph.raw_cuda_graph())
    _, count = check(cu.cuGraphGetNodes(handle, 0))
    nodes, actual = check(cu.cuGraphGetNodes(handle, count))
    return [node for node in nodes[:actual]
            if check(cu.cuGraphNodeGetType(node)) == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL]


def graph_windows(graph):
    result = []
    for node in graph_kernels(graph):
        value = check(cu.cuGraphKernelNodeGetAttribute(
            node, cu.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW))
        w = value.accessPolicyWindow
        result.append([int(w.num_bytes), float(w.hitRatio), int(w.hitProp),
                       int(w.base_ptr), int(w.missProp)])
    return result


def explicit_graph_window(graph, tensor, fraction):
    value = cu.CUkernelNodeAttrValue()
    w = value.accessPolicyWindow
    w.base_ptr = tensor.data_ptr()
    w.num_bytes = tensor.numel() * tensor.element_size()
    w.hitRatio = fraction
    w.hitProp = cu.CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING
    w.missProp = cu.CUaccessProperty.CU_ACCESS_PROPERTY_NORMAL
    for node in graph_kernels(graph):
        check(cu.cuGraphKernelNodeSetAttribute(
            node, cu.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boot', type=int, required=True)
    parser.add_argument('--trials', type=int, default=31)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    torch.cuda.init()
    assert torch.cuda.get_device_capability() == (12, 1)
    sources = json.loads(Path(__file__).with_name('source-hashes.json').read_text())
    package = Path(vllm.__file__).resolve().parent.parent
    for name, expected in sources.items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected, name
    spec = importlib.util.spec_from_file_location(
        'prior_probe', Path(__file__).parents[1] /
        '2026-10-25-moe-overlap/bundles/l2-policy/l2_policy_probe.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    dev = check(cu.cuDeviceGet(0))
    ctx = check(cu.cuDevicePrimaryCtxRetain(dev))
    check(cu.cuCtxSetCurrent(ctx))
    read = prior.read_kernel('default')
    pref = l2_prefetch_kernel(2, 128, 4096)
    check(cu.cuCtxSetCurrent(ctx))
    limit = cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE
    original = int(check(cu.cuCtxGetLimit(limit)))
    maximum = int(check(cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, dev)))
    maximum_window = int(check(cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, dev)))
    assert maximum > 0 and maximum_window > 0
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())
    key = 'mtp.2.markov_head.head.weight'
    source = args.checkpoint / index['weight_map'][key]
    result = {'boot': args.boot, 'status': 'running', 'original_limit': original,
              'maximum_limit': maximum, 'maximum_window': maximum_window,
              'source_hashes': sources, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': [], 'started': time.time()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    save()
    stream = torch.cuda.Stream()
    interference = torch.empty(96 << 20, device='cuda', dtype=torch.uint8)
    interference.fill_(41)
    stream_segs = torch.tensor([interference.data_ptr(), interference.numel()],
                               device='cuda', dtype=torch.int64)
    sink = torch.empty(prior.GRID * prior.BLOCK, device='cuda', dtype=torch.uint32)
    flush = torch.empty(64 << 20, device='cuda', dtype=torch.uint8)
    after = torch.empty(16 << 20, device='cuda', dtype=torch.uint8)
    after.fill_(17)
    after_segs = torch.tensor([after.data_ptr(), after.numel()], device='cuda', dtype=torch.int64)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rng = random.Random(411 + args.boot)
    try:
        for tp in (3, 4):
            n = ((129280 + tp * 64 - 1) // (tp * 64)) * 64
            with safe_open(str(source), framework='pt', device='cpu') as f:
                raw = f.get_slice(key)[:n, :].clone()
            assert raw.shape == (n, 256) and raw.dtype == torch.bfloat16
            weight_hash = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()
            packed = pack_bf16_weight(raw.to('cuda'))
            config = tile_config(16, 256)
            partial = packed_vocab_markov_greedy_partials(
                n, 256, packed.exponent_base, block_M=16, out_dtype=T.float32, **config)
            tiles = (n + config['block_N'] - 1) // config['block_N']
            finish = finish_vocab_greedy_pairs(tiles, pair_width=4, top2=True)
            check(cu.cuCtxSetCurrent(ctx))
            b = packed.values.numel()
            assert b <= maximum_window
            segs = torch.tensor([packed.values.data_ptr(), b], device='cuda', dtype=torch.int64)
            # Window protects only the packed payload: no copies or arena relocation.
            # Row offsets and exceptions stay at their ordinary native addresses.
            arms = [('driver_default', original, False, 0),
                    ('driver_null', original, False, 0),
                    ('prefetch_default', original, False, 1),
                    ('prefetch_default_each', original, False, 2),
                    ('window_default', original, True, 0),
                    ('both_default', original, True, 1),
                    ('demand', 0, False, 0), ('null', 0, False, 0),
                    ('prefetch_once', 0, False, 1), ('prefetch_each', 0, False, 2),
                    ('partition_small', 1 << 20, False, 0),
                    ('window_small', 1 << 20, True, 0), ('both_small', 1 << 20, True, 1),
                    ('partition_fit', min(b, maximum), False, 0),
                    ('window_fit', min(b, maximum), True, 0),
                    ('both_fit', min(b, maximum), True, 1),
                    ('contig_default', original, 'contig', 0),
                    ('both_contig_default', original, 'contig', 1),
                    ('pos0_default', original, 'pos0', 0),
                    ('base_stream_default', original, 'base_stream', 0),
                    ('both_base_stream_default', original, 'base_stream', 1),
                    ('contig_mid', 8 << 20, 'contig', 0),
                    ('both_contig_mid', 8 << 20, 'contig', 1),
                    ('pos0_fit', min(b, maximum), 'pos0', 0)]
            for rows in (1, 8, 16):
                xs = [torch.randn(rows, 256, device='cuda', dtype=torch.bfloat16) for _ in range(5)]
                base = torch.randn(rows, 5, n, device='cuda', dtype=torch.float32)
                vals = torch.empty(rows, tiles, device='cuda', dtype=torch.float32)
                ids = torch.empty(rows, tiles, device='cuda', dtype=torch.int32)
                outputs = [torch.empty(rows, 4, device='cuda', dtype=torch.float32) for _ in range(5)]
                for stress in (False, True):
                    graphs = {}
                    attrs = {}
                    captured_attrs = {}
                    reference = None
                    for name, budget, uses_window, prefetch in arms:
                        torch.cuda.synchronize()
                        check(cu.cuCtxSetLimit(limit, budget))
                        actual = int(check(cu.cuCtxGetLimit(limit)))
                        check(cu.cuCtxResetPersistingL2Cache())
                        window(stream, packed.values if uses_window is True else None, min(1.0, actual / b))

                        def apply_phase_window(pos):
                            if uses_window == 'contig':
                                window(stream, packed.values, 1.0, num_bytes=min(b, actual))
                            elif uses_window == 'pos0':
                                if pos == 0:
                                    window(stream, packed.values, min(1.0, actual / b), num_bytes=b)
                                else:
                                    window(stream)
                            elif uses_window == 'base_stream':
                                window(stream, base, 1.0, hit_prop=cu.CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING)

                        def clear_phase_window():
                            if uses_window in ('contig', 'pos0', 'base_stream'):
                                window(stream)

                        def step():
                            if prefetch == 1:
                                pref(segs)
                            for pos in range(5):
                                if prefetch == 2:
                                    pref(segs)
                                apply_phase_window(pos)
                                partial(xs[pos], packed.values, packed.exception_rows,
                                        packed.exception_entries, packed.row_offsets,
                                        base[:, pos], vals, ids, n)
                                clear_phase_window()
                                finish(vals, ids, outputs[pos], 0)
                                if stress and pos != 4:
                                    read(stream_segs, sink)

                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            step(); step()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph, stream=stream):
                            step()
                        captured_attrs[name] = graph_windows(graph)
                        if uses_window is True and any(a[0] != b for a in captured_attrs[name]):
                            explicit_graph_window(graph, packed.values, min(1.0, actual / b))
                        attrs[name] = graph_windows(graph)
                        assert attrs[name], name
                        if isinstance(uses_window, bool):
                            assert all(a[0] == (b if uses_window else 0) for a in attrs[name]), (name, attrs[name])
                            if uses_window:
                                expected_ratio = min(1.0, actual / b)
                                assert all(abs(a[1] - expected_ratio) < 1e-6
                                           and a[2] == int(cu.CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING)
                                           and a[3] == packed.values.data_ptr()
                                           and a[4] == int(cu.CUaccessProperty.CU_ACCESS_PROPERTY_NORMAL)
                                           for a in attrs[name]), (name, attrs[name])
                        elif uses_window == 'contig':
                            active = [a for a in attrs[name] if a[0] > 0]
                            assert len(active) == 5 and all(a[0] == min(b, actual) and abs(a[1] - 1.0) < 1e-6 for a in active), (name, attrs[name])
                        elif uses_window == 'pos0':
                            active = [a for a in attrs[name] if a[0] > 0]
                            assert len(active) == 1 and active[0][0] == b, (name, attrs[name])
                        elif uses_window == 'base_stream':
                            active = [a for a in attrs[name] if a[0] > 0]
                            assert len(active) == 5 and all(a[0] == base.numel() * base.element_size() and a[2] == int(cu.CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING) for a in active), (name, attrs[name])
                        graph.instantiate()
                        window(stream)
                        graph.replay()
                        torch.cuda.synchronize()
                        copies = [t.cpu() for t in outputs]
                        if reference is None:
                            reference = copies
                        assert all(torch.equal(a.view(torch.uint8), c.view(torch.uint8))
                                   for a, c in zip(reference, copies)), name
                        graphs[name] = graph
                    samples = []
                    for trial in range(args.trials):
                        order = list(arms)
                        rng.shuffle(order)
                        for name, budget, _, _ in order:
                            torch.cuda.synchronize()
                            check(cu.cuCtxSetLimit(limit, budget))
                            applied = int(check(cu.cuCtxGetLimit(limit)))
                            check(cu.cuCtxResetPersistingL2Cache())
                            flush.zero_()
                            start.record(); graphs[name].replay(); end.record()
                            end.synchronize()
                            us = start.elapsed_time(end) * 1000
                            # Default demand reuse immediately after the draft region,
                            # including any stale persisting lines. This is a cache probe,
                            # not the required full-path prefill coexistence benchmark.
                            read(after_segs, sink)
                            start.record(); read(after_segs, sink); end.record()
                            end.synchronize()
                            after_us = start.elapsed_time(end) * 1000
                            samples.append({'trial': trial, 'arm': name, 'us': us,
                                            'after_us': after_us, 'limit_bytes': applied})
                    case = {'tp': tp, 'rows': rows, 'expert_stream_stress': stress,
                            'weight_sha256': weight_hash, 'payload_bytes': b,
                            'exception_count': packed.exceptions, 'graph_attributes': attrs,
                            'captured_graph_attributes': captured_attrs,
                            'all_bits_equal': True, 'samples': samples}
                    case['median_us'] = {a[0]: statistics.median(s['us'] for s in samples if s['arm'] == a[0]) for a in arms}
                    result['cases'].append(case)
                    save()
                    print(json.dumps({k: v for k, v in case.items() if k in ('tp', 'rows', 'expert_stream_stress', 'median_us')}), flush=True)
                    for graph in graphs.values():
                        graph.reset()
                    del graphs
            del packed
        result['status'] = 'passed'
    except BaseException as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        torch.cuda.synchronize()
        window(stream)
        check(cu.cuCtxResetPersistingL2Cache())
        check(cu.cuCtxSetLimit(limit, original))
        result['restored_limit'] = int(check(cu.cuCtxGetLimit(limit)))
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
