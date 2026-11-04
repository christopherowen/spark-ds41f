"""Balanced GPU measurements and a zero-margin, per-case performance gate."""

import math
import random
import statistics


def bound(log_ratios):
    """A predeclared five-standard-error upper bound; no time tolerance."""
    mean = statistics.fmean(log_ratios)
    sem = statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
    return dict(
        mean_ratio=math.exp(mean),
        upper_ratio=math.exp(mean + 5 * sem),
        lower_ratio=math.exp(mean - 5 * sem),
        rounds=len(log_ratios),
    )


def classify(samples, identical_program=False):
    summary = bound(samples)
    if identical_program:
        status = "equal_program"
    elif summary["upper_ratio"] <= 1:
        status = "better"
    else:
        status = "not_proved"
    return dict(summary, status=status)


def measure(launches, flush, seed, expected, rounds=40):
    import torch

    graphs, outputs = {}, {}
    for name, launch in launches.items():
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = launch()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            outputs[name].view(torch.int32), expected.view(torch.int32)
        ), name
        graphs[name] = graph
    # Equal warmup for every role, before collecting any sample.
    for _ in range(8):
        for graph in graphs.values():
            flush.zero_()
            graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    rng = random.Random(seed)
    pairs = []
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        # A role and its mirror cancel a linear drift within each round.
        order += order[::-1]
        row = {name: [] for name in graphs}
        for name in order:
            flush.zero_()
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            row[name].append(start.elapsed_time(end) * 1000)
        pairs.append(dict(order=order, times_us=row))
    logs = {
        name: [
            math.log(
                statistics.fmean(row["times_us"][name])
                / statistics.fmean(row["times_us"]["reference"])
            )
            for row in pairs
        ]
        for name in graphs
        if name != "reference"
    }
    return dict(pairs=pairs, log_ratios=logs), outputs
