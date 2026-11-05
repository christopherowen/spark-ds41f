"""Paired CUPTI kernel durations with the original CUDA event ordering markers.

External graph events add about 3 us to the ~1 us tail kernels. Keep those raw
results; use hardware activity timestamps for compute and retain event intervals too.
"""
import hashlib
import json
import math
import random
import statistics


def measure_activity(launches, flush, seed, expected, out, rounds=120, samples=64):
    import torch

    graphs, events = {}, {}
    for role, launch in launches.items():
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        events[role] = [(torch.cuda.Event(enable_timing=True, external=True),
                         torch.cuda.Event(enable_timing=True, external=True))
                        for _ in range(samples)]
        with torch.cuda.graph(graph):
            for begin, end in events[role]:
                flush.zero_()
                begin.record()
                actual = launch()
                end.record()
        graph.replay()
        torch.cuda.synchronize()
        values = actual if isinstance(actual, tuple) else (actual,)
        checks = expected if isinstance(expected, tuple) else (expected,)
        assert all(torch.equal(v.view(torch.uint8), c.view(torch.uint8))
                   for v, c in zip(values, checks, strict=True)), role
        graphs[role] = graph
    for _ in range(8):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    rng = random.Random(seed)
    orders, event_intervals = [], []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(rounds):
            order = list(graphs)
            rng.shuffle(order)
            order += order[::-1]
            orders.append(order)
            intervals = {role: [] for role in launches}
            for role in order:
                graphs[role].replay()
                events[role][-1][1].synchronize()
                intervals[role].append([begin.elapsed_time(end) * 1000
                                        for begin, end in events[role]])
            event_intervals.append(intervals)
    profile.export_chrome_trace(str(out))
    trace = json.loads(out.read_text())
    all_kernels = sorted((e for e in trace['traceEvents'] if e.get('cat') == 'kernel'),
                         key=lambda e: e['ts'])
    expected_count = rounds * len(launches) * 2 * samples
    assert len(all_kernels) == 2 * expected_count, (len(all_kernels), samples)
    flush_kernels, kernels = all_kernels[::2], all_kernels[1::2]
    # zero_ is an ATen fill in this image. Retain and validate it separately:
    # the operation timer, like the original event interval, starts after flush.
    assert all('FillFunctor<unsigned char>' in e['name'] for e in flush_kernels)
    assert all('FillFunctor<unsigned char>' not in e['name'] for e in kernels)
    assert len({e['args']['stream'] for e in kernels}) == 1
    cursor, pairs, names = 0, [], {}
    for order in orders:
        times = {role: [] for role in launches}
        for role in order:
            group = kernels[cursor:cursor + samples]
            cursor += samples
            assert len({e['name'] for e in group}) == 1
            name = group[0]['name']
            assert names.setdefault(role, name) == name
            if role in ('reference', 'null'):
                assert 'at::native' in name, name
            elif role == 'candidate':
                assert 'collectives_rank_rows_small' in name or 'collectives_rank_rows_bytes' in name, name
            else:
                assert 'collectives_rank_rows_bytes' in name, name
            values = [e['dur'] for e in group]
            assert all(math.isfinite(v) and v > 0 for v in values)
            times[role].append(values)
        pairs.append(dict(order=order, times_us=times))
    logs = {role: [math.log(statistics.fmean(t for a in r['times_us'][role] for t in a)
                           / statistics.fmean(t for a in r['times_us']['reference'] for t in a))
                   for r in pairs] for role in launches if role != 'reference'}
    return dict(pairs=pairs, log_ratios=logs, samples_per_replay=samples,
                timer='CUPTI GPU activity start/end, original external-event ordering retained',
                event_intervals_us=event_intervals,
                activity_kernels=len(kernels), role_kernels=names,
                cache_flush=dict(count=len(flush_kernels),
                    kernels=sorted({e['name'] for e in flush_kernels}),
                    mean_duration_us=statistics.fmean(e['dur'] for e in flush_kernels)),
                trace_sha256=hashlib.sha256(out.read_bytes()).hexdigest()), None
