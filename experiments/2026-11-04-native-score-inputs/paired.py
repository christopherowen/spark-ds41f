"""Balanced GPU measurements and a zero-margin, per-case performance gate."""

import math
import random
import statistics
from collections import Counter


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


def extend(primary, additional):
    """Extra precision may resolve uncertainty, never erase an observed increase."""
    first = classify(primary)
    followup = classify(additional)
    pooled = classify(primary + additional)
    return dict(
        primary=first,
        extension=followup,
        pooled=pooled,
        passed=first["mean_ratio"] <= 1
        and followup["status"] == "better"
        and pooled["status"] == "better",
    )


def audit(row, rounds):
    """Recompute a case's ratios and decision from all its raw observations."""
    records = row["measured"]["pairs"]
    assert len(records) == rounds
    roles = ("reference", "candidate", "previous", "null")
    logs = {role: [] for role in roles if role != "reference"}
    rng = random.Random(row["data"]["sample_order_seed"])
    for record in records:
        order = list(roles)
        rng.shuffle(order)
        order += order[::-1]
        assert record["order"] == order and Counter(order) == dict.fromkeys(roles, 2)
        averages = {}
        assert set(record["times_us"]) == set(roles)
        for role, samples in record["times_us"].items():
            assert len(samples) == 2
            if isinstance(samples[0], list):
                assert all(len(batch) == 8 for batch in samples)
                samples = [t for batch in samples for t in batch]
            assert all(math.isfinite(t) and t > 0 for t in samples)
            averages[role] = statistics.fmean(samples)
        for role in logs:
            logs[role].append(math.log(averages[role] / averages["reference"]))
    for role, values in logs.items():
        assert all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
            for a, b in zip(values, row["measured"]["log_ratios"][role], strict=True)
        )
    same = (
        row["data"]["reference_preparation_aliases_input"]
        and row["data"]["shared_output_address"]
        and row["cuda_source_sha256"]["candidate"]
        == row["cuda_source_sha256"]["reference"]
    )
    assert same == row["same_program"]
    stats = classify(logs["candidate"], identical_program=same)
    assert stats.keys() == row["gate"].keys()
    for key, value in stats.items():
        if isinstance(value, float):
            # ARM/x86 exp() can differ by an ULP; the decision remains exact.
            assert math.isclose(value, row["gate"][key], rel_tol=1e-14)
        else:
            assert value == row["gate"][key]
    return logs, stats


def measure(launches, flush, seed, expected, rounds=120, samples_per_replay=8):
    import torch

    graphs, outputs, events = {}, {}, {}
    for name, launch in launches.items():
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Explicit event nodes remove Python submission gaps from the interval.
        # Each sample starts cold, including samples within the same replay.
        events[name] = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(samples_per_replay)
        ]
        with torch.cuda.graph(graph):
            for start, end in events[name]:
                flush.zero_()
                start.record()
                outputs[name] = launch()
                end.record()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            outputs[name].view(torch.int32), expected.view(torch.int32)
        ), name
        graphs[name] = graph
    # Equal warmup for every role, before collecting any sample.
    for _ in range(8):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    rng = random.Random(seed)
    pairs = []
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        # A role and its mirror cancel a linear drift within each round.
        order += order[::-1]
        row = {name: [] for name in graphs}
        for name in order:
            graphs[name].replay()
            events[name][-1][1].synchronize()
            row[name].append(
                [start.elapsed_time(end) * 1000 for start, end in events[name]]
            )
        pairs.append(dict(order=order, times_us=row))
    logs = {
        name: [
            math.log(
                statistics.fmean(t for batch in row["times_us"][name] for t in batch)
                / statistics.fmean(
                    t for batch in row["times_us"]["reference"] for t in batch
                )
            )
            for row in pairs
        ]
        for name in graphs
        if name != "reference"
    }
    return dict(
        pairs=pairs,
        log_ratios=logs,
        timer="external event nodes inside the graph; no host submission gap",
        samples_per_replay=samples_per_replay,
    ), outputs
