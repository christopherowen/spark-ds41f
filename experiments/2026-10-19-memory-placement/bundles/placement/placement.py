"""Within-process buffer placement probe, on a node with no serving container."""
import json
import math
import random
import statistics
from pathlib import Path


def memory_plan(available, count=20, buffer_bytes=2 << 30, guard=3 << 30):
    if count < 2 or buffer_bytes < 4 or buffer_bytes % 4:
        raise ValueError("need two or more word-aligned buffers")
    reserve = guard + (2 << 30)  # compiler and result workspace, above the guard
    if available - count * buffer_bytes < reserve:
        raise RuntimeError("placement buffers would cross the memory reserve")
    return dict(count=count, buffer_bytes=buffer_bytes, guard=guard, reserve=reserve)


def mem_available():
    fields = dict(line.split(":", 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def describe(values):
    mean = statistics.fmean(values)
    return dict(mean_ms=mean, ci95_ms=1.984 * statistics.stdev(values) / math.sqrt(len(values)),
                min_ms=min(values), max_ms=max(values), samples_ms=values)


def main():
    import tilelang
    import tilelang.language as T
    import torch

    plan = memory_plan(mem_available())
    block = 4096

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def read_kernel(chunk_words: int):
        words, selections = T.dynamic('words, selections')
        partials = T.dynamic('partials')
        @T.prim_func
        def placement_read(source: T.Tensor((words,), T.int32),
                           chunks: T.Tensor((selections,), T.int32),
                           out: T.Tensor((partials,), T.int32)):
            with T.Kernel(T.ceildiv(chunk_words, block), selections, threads=256) as (bx, by):
                values = T.alloc_fragment((block,), T.int32)
                total = T.alloc_fragment((1,), T.int32)
                for i in T.Parallel(block):
                    offset = bx * block + i
                    address = T.cast(chunks[by], T.int64) * chunk_words + offset
                    values[i] = T.if_then_else((offset < chunk_words) & (address < words), source[address], 0)
                T.reduce_sum(values, total, dim=0)
                out[by * T.ceildiv(chunk_words, block) + bx] = total[0]
        return placement_read

    def read(source, chunks, chunk_words, output):
        read_kernel(chunk_words)(source, chunks, output)

    # Exercise partial tiles, permutation and >1 chunk before allocating the probe.
    for chunk_words in (17, 4097):
        source = torch.ones(chunk_words * 5, dtype=torch.int32, device='cuda')
        chunks = torch.tensor([4, 0, 2], dtype=torch.int32, device='cuda')
        output = torch.empty(3 * math.ceil(chunk_words / block), dtype=torch.int32, device='cuda')
        read(source, chunks, chunk_words, output)
        assert output.sum().item() == 3 * chunk_words
    del source, chunks, output

    buffers = []
    minimum = mem_available()
    for index in range(plan['count']):
        if mem_available() < plan['buffer_bytes'] + plan['reserve']:
            raise RuntimeError(f'memory reserve before buffer {index}')
        buffers.append(torch.ones(plan['buffer_bytes'] // 4, dtype=torch.int32, device='cuda'))
        torch.cuda.synchronize()
        minimum = min(minimum, mem_available())
        if minimum < plan['guard']:
            raise RuntimeError('memory guard breached after allocation')
    words = plan['buffer_bytes'] // 4
    expert_words = 18_800_000 // 4
    rng = random.Random(41)
    selections = rng.sample(range(words // expert_words), 6)
    modes = [('stream', words, [0]), ('expert6', expert_words, selections)]
    result = dict(plan=plan, image='r6x', physical_layout='unavailable through CUDA allocator',
                  minimum_mem_available_bytes=minimum, buffers=[dict(index=i, virtual_address=b.data_ptr())
                  for i, b in enumerate(buffers)], modes={})
    for name, chunk_words, selected in modes:
        chunks = torch.tensor(selected, dtype=torch.int32, device='cuda')
        output = torch.empty(len(selected) * math.ceil(chunk_words / block), dtype=torch.int32, device='cuda')
        read_kernel(chunk_words)  # compile outside event measurements
        for source in buffers:
            for _ in range(3):
                read(source, chunks, chunk_words, output)
            assert output.sum().item() == len(selected) * chunk_words
        samples = [[] for _ in buffers]
        for repeat in range(100):
            order = list(range(len(buffers)))
            rng.shuffle(order)
            for index in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                read(buffers[index], chunks, chunk_words, output)
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end))
            minimum = min(minimum, mem_available())
            if minimum < plan['guard']:
                raise RuntimeError('memory guard breached during measurement')
        rows = [describe(s) for s in samples]
        means = [row['mean_ms'] for row in rows]
        mid = statistics.fmean(means)
        result['modes'][name] = dict(bytes_per_read=len(selected) * chunk_words * 4,
                                    selected_chunks=selected, buffers=rows,
                                    spread_pct=100 * (max(means) - min(means)) / mid,
                                    max_deviation_pct=max(100 * abs(v / mid - 1) for v in means))
    result['minimum_mem_available_bytes'] = minimum
    print(json.dumps(result), flush=True)
    print(json.dumps({'bits': 'placement partial-read checksums', 'differs_at': []}))


if __name__ == '__main__':
    main()
