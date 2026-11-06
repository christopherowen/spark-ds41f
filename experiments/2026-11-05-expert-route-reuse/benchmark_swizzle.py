"""Paired GPU screening of the whole routed-MoE call; synthetic inputs only.

Every arm shares prepared weights, activations, route IDs/weights, workspace and
output storage. Panel 0 is an unreleased control on the same source commit.
A process repeat is not a serving boot; no serving-performance claim is made.
"""
import argparse
import hashlib
import importlib.util
import sys
import json
from pathlib import Path
import statistics
import subprocess

import torch
import tile_kernels
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang import moe
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
from screen_statistics import arm_order, interval

E, TOPK, H = 384, 6, 5120
PANELS = (0, 2, 3, 4, 8, 16)
SAMPLES, REPLAYS, CALLS = 8, 6, 3


def emit(kind, **values):
    print(json.dumps({'kind': kind, **values}), flush=True)


def gpu_state():
    command = ['nvidia-smi', '--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.gr,clocks.mem,power.draw,power.limit', '--format=csv,noheader,nounits']
    p = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    return {'nvidia_smi': p.stdout.strip(),
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def route_ids(rows, routing, generator):
    if routing == 'disjoint':
        assert rows * TOPK <= E
        ids = torch.arange(rows * TOPK, device='cuda').reshape(rows, TOPK)
    elif routing == 'independent-groups':
        assert rows == 48
        groups = torch.arange(8, device='cuda').repeat_interleave(6)
        ids = groups[:, None] * TOPK + torch.arange(TOPK, device='cuda')
    elif routing == 'balanced':
        ids = torch.arange(rows * TOPK, device='cuda').reshape(rows, TOPK) % E
    elif routing == 'prefill-m64':
        ids = torch.arange(rows * TOPK, device='cuda').reshape(rows, TOPK) % 8
    else:
        scores = torch.rand((rows, E), generator=generator, device='cuda')
        if routing == 'skewed':
            rank = torch.arange(1, E + 1, device='cuda', dtype=torch.float32)
            scores = -0.5 * rank.log() - torch.log(-torch.log(scores.clamp_min(1e-20)))
        ids = scores.topk(TOPK, dim=-1).indices
    return ids.to(torch.int32).contiguous()


def weights(plan):
    gen = torch.Generator(device='cuda').manual_seed(711)
    prepared = []
    for n, k in ((2 * plan.intermediate, H), (H, plan.intermediate)):
        packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8,
                               generator=gen, device='cuda')
        scales = torch.randint(118, 122, (E, n, k // 32), dtype=torch.uint8,
                               generator=gen, device='cuda')
        # Cover both exact scale encodings without expanding full FP32 weights.
        scales[1::8, 0, 0] = 118
        scales[1::8, 0, 1] = 123
        prepared.append(plan.prepare_weight(packed, scales))
        del packed, scales
    return tuple(prepared)


def graph(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(); call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            call()
    return capture


def elapsed(capture):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); capture.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000 / CALLS


def case(plans, prepared, rows, routing, tp, process_id):
    for p in plans.values():
        if routing == 'prefill-m64':
            p.tile_height = lambda r, _p=p: 64 if r == rows else moe.MoeExpertsV41Plan.tile_height(_p, r)
        elif 'tile_height' in p.__dict__:
            del p.__dict__['tile_height']
    workspace_sizes = {arm: moe.workspace_bytes(p.workspace_specs(rows, torch.int32))
                       for arm, p in plans.items()}
    assert len(set(workspace_sizes.values())) == 1, ('workspace changed', workspace_sizes)
    workspace = torch.empty(workspace_sizes[0], dtype=torch.uint8, device='cuda')
    times = {p: [] for p in plans}
    route_hashes = []
    for sample in range(SAMPLES):
        seed = 1000 * rows + sample
        gen = torch.Generator(device='cuda').manual_seed(seed)
        ids = route_ids(rows, routing, gen)
        route_weights = (torch.rand((rows, TOPK), generator=gen, device='cuda') + .1).contiguous()
        x = (torch.randn((rows, H), generator=gen, device='cuda') * .5).bfloat16()
        tokens = cast_mxfp8(x)
        out = torch.empty_like(x)
        saved_ids, saved_weights = ids.clone(), route_weights.clone()
        route_hashes.append(hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
        captures = {}
        expected = None
        for panel, plan in plans.items():
            def call(plan=plan):
                plan.run(tokens, ids, route_weights, prepared, out, workspace)
            call(); torch.cuda.synchronize()
            if panel == 0:
                expected = out.clone()
                assert torch.isfinite(expected).all().item(), 'non-finite control output'
            else:
                assert torch.equal(expected.view(torch.int16), out.view(torch.int16)), (tp, rows, routing, sample, panel, 'output mismatch')
            # Compare the same first 17 rows in a small invocation (M16) too.
            if panel == 0:
                tiny = torch.empty_like(x[:17])
                plan.run(cast_mxfp8(x[:17].contiguous()), ids[:17], route_weights[:17],
                         prepared, tiny, workspace)
                torch.cuda.synchronize()
                assert torch.equal(expected[:17].view(torch.int16), tiny.view(torch.int16)), (
                    tp, rows, routing, sample, 'batch-size output mismatch')
                del tiny
            captures[panel] = graph(call)
        assert torch.equal(ids, saved_ids) and torch.equal(route_weights, saved_weights), 'routes mutated'
        raw = {p: [] for p in plans}
        panels = list(plans)
        for repeat in range(REPLAYS):
            order = arm_order(panels, sample * REPLAYS + repeat)
            for panel in order:
                raw[panel].append(elapsed(captures[panel]))
        medians = {p: statistics.median(v) for p, v in raw.items()}
        for panel in plans:
            times[panel].append(medians[panel])
        emit('sample', process_id=process_id, tp_projection=tp, rows=rows,
             routing=routing, evidence_kind='synthetic', sample=sample, seed=seed,
             route_sha256=route_hashes[-1], bit_equal=True, routes_unchanged=True,
             first_17_rows_batch_invariant=True,
             raw_us=raw, median_us=medians)
        del captures, expected, tokens, out, x, ids, route_weights, saved_ids, saved_weights
    result = []
    for panel in plans:
        differences = [b-a for a, b in zip(times[0], times[panel])]
        ratios = [100*(b/a-1) for a, b in zip(times[0], times[panel])]
        result.append({'arm': panel, 'before_us': statistics.fmean(times[0]),
                       'after_us': statistics.fmean(times[panel]),
                       'difference_us': statistics.fmean(differences),
                       'difference_95_interval_us': interval(differences),
                       'change_percent': statistics.fmean(ratios),
                       'change_95_interval_percent': interval(ratios)})
    emit('case', process_id=process_id, tp_projection=tp, rows=rows, routing=routing,
         evidence_kind='synthetic', tile_height=plans[0].tile_height(rows),
         route_hashes=route_hashes, workspace_bytes=workspace_sizes, comparisons=result, gpu=gpu_state())
    del workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-id', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--default-comparison', action='store_true')
    parser.add_argument('--control-source-sha256')
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(moe.__file__).read_bytes()).hexdigest()
    assert source_hash == args.source_sha256, 'source differs from the published candidate'
    control = None
    if args.default_comparison:
        path = Path('/b/control_moe.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == args.control_source_sha256
        spec = importlib.util.spec_from_file_location(
            'vllm.models.deepseek_v4_1.tilelang._control_moe', path)
        control = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = control
        spec.loader.exec_module(control)
    torch.set_num_threads(2)
    emit('identity', process_id=args.process_id, source_commit=args.source_commit,
         source_sha256=source_hash, torch=torch.__version__, cuda=torch.version.cuda,
         cache_policy='two warmups per arm; three repeated calls per CUDA graph; no explicit L2 flush',
         samples=SAMPLES, replays=REPLAYS, calls=CALLS, control=('unreleased candidate-38 default' if control else 'unreleased same-commit panel 0'),
         timing_scope='whole routed-MoE plan, one GPU; not serving throughput or physical TP topology',
         arms=({'0': 'untouched candidate-38 default', '1': 'fixed default'}
               if args.default_comparison else {str(p): 'explicit panel ' + str(p) for p in PANELS}),
         control_source_commit=('1f33102547e4da5c64712390604695d1abd577c4' if control else args.source_commit),
         control_source_sha256=args.control_source_sha256,
         gpu=gpu_state())
    with set_current_vllm_config(VllmConfig()):
        for tp in ((4,) if args.smoke else (3, 4)):
            inter = 2304 // tp
            panels = (0, 2, 3) if args.smoke else PANELS
            if control is None:
                plans = {panel: moe.MoeExpertsV41Plan(E, TOPK, H, inter, swiglu_limit=10.,
                                                    prefill_swizzle_panel=panel) for panel in panels}
            else:
                plans = {0: control.MoeExpertsV41Plan(E, TOPK, H, inter, swiglu_limit=10.),
                         1: moe.MoeExpertsV41Plan(E, TOPK, H, inter, swiglu_limit=10.)}
            for plan in plans.values():
                if control is None:
                    plan.prefill_dispatch[torch.int32].min_routes = 0
                plan.declare_kernels()
            if control is not None:
                generated = []
                for name, (factory, call_args, kwargs) in plans[0].kernels().items():
                    old = factory(*call_args, **kwargs).get_kernel_source()
                    new_factory, new_args, new_kwargs = plans[1].kernels()[name]
                    new = new_factory(*new_args, **new_kwargs).get_kernel_source()
                    assert old == new, ('default compiled source differs', tp, name)
                    assert factory.pass_configs == new_factory.pass_configs
                    generated.append({'kernel': name, 'sha256': hashlib.sha256(old.encode()).hexdigest()})
                emit('default_native_compilation', tp_projection=tp,
                     equal_generated_sources=generated, compilation_settings_equal=True)
            tile_kernels.config.set_token_alignment(32)
            prepared = weights(plans[0])
            points = [(2048, 'uniform')] if args.smoke else (
                [(48, 'disjoint'), (48, 'independent-groups'), (96, 'uniform')]
                + [(r, 'uniform') for r in (1023, 1024, 2047)]
                + [(r, routing) for r in (2048, 4096, 8192, 16384) for routing in ('uniform', 'skewed')]
                + [(8192, 'balanced')]
                if control is not None else
                [(48, 'disjoint'), (48, 'independent-groups'), (96, 'uniform')]
                + [(r, 'prefill-m64') for r in (171, 500)]
                + [(r, 'uniform') for r in (171, 500, 1023, 1024, 2047)]
                + [(r, routing) for r in (2048, 4096, 6144, 8192, 16384) for routing in ('uniform', 'skewed')]
                + [(8192, 'balanced')])
            for rows, routing in points:
                case(plans, prepared, rows, routing, tp, args.process_id)
            del prepared, plans
            torch.cuda.empty_cache()
    emit('complete', process_id=args.process_id, status='pass', gpu=gpu_state())


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        try:
            state = gpu_state()
        except Exception as state_error:
            state = {'unavailable': repr(state_error)}
        emit('failure', error_type=type(exc).__name__, message=str(exc), gpu=state)
        raise
