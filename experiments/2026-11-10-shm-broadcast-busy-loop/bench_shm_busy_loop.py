#!/usr/bin/env python3
"""Benchmark MessageQueue SpinCondition busy_loop_s=1.0s (baseline) vs 0.002s (candidate).

Measures a two-process local MessageQueue (n_reader=1, n_local_reader=1, matching
rank-0 EngineCore <-> Worker_TP0 worker_response_mq / rpc_broadcast_mq) across:
  1. decode_step_21ms: 200 simulated 21 ms GPU decode steps
  2. prefill_chunk_250ms: 20 simulated 250 ms GPU prefill chunks
  3. sub_ms_rpc_burst_0p2ms: 500 back-to-back RPCs spaced 0.2 ms apart (< 2 ms)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import statistics
import time
from typing import Any

import vllm.distributed.device_communicators.shm_broadcast as shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue


def _cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return float(ru.ru_utime + ru.ru_stime)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def _writer_entry(
    handle: Any,
    ready_q: mp.Queue,
    go_q: mp.Queue,
    n_steps: int,
    step_interval_s: float,
) -> None:
    # Writer is created in parent; wait for reader ready signal
    pass


def _reader_worker(
    handle: Any,
    busy_loop_s: float,
    n_warmup: int,
    n_steps: int,
    ready_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    reader = MessageQueue.create_from_handle(handle, rank=0)
    try:
        reader.wait_until_ready()
        reader._spin_condition.busy_loop_s = busy_loop_s

        yield_count = 0
        poll_count = 0
        orig_yield = shm_broadcast.sched_yield
        orig_poll = reader._spin_condition.poller.poll

        def counting_yield() -> None:
            nonlocal yield_count
            yield_count += 1
            orig_yield()

        def counting_poll(*args: Any, **kwargs: Any) -> Any:
            nonlocal poll_count
            poll_count += 1
            return orig_poll(*args, **kwargs)

        shm_broadcast.sched_yield = counting_yield
        reader._spin_condition.poller.poll = counting_poll  # type: ignore[method-assign]

        ready_q.put("ready")

        for _ in range(n_warmup):
            msg = reader.dequeue(timeout=10)
            assert msg[0] == "warmup"

        # Drain any pending notify from warmup so measurement starts cleanly
        if reader._spin_condition.local_notify_socket.poll(timeout=0):
            reader._spin_condition.local_notify_socket.recv(
                flags=shm_broadcast.zmq.NOBLOCK, copy=False
            )
        reader._spin_condition.last_read = time.monotonic()

        yield_count = 0
        poll_count = 0
        latencies_us: list[float] = []

        t_wall_0 = time.monotonic()
        t_cpu_0 = _cpu_seconds()

        for expected_idx in range(n_steps):
            kind, idx, send_ns, payload = reader.dequeue(timeout=10)
            recv_ns = time.monotonic_ns()
            assert kind == "step" and idx == expected_idx and len(payload) == 64
            latencies_us.append((recv_ns - send_ns) / 1000.0)

        t_cpu_1 = _cpu_seconds()
        t_wall_1 = time.monotonic()

        shm_broadcast.sched_yield = orig_yield

        wall_s = max(1e-9, t_wall_1 - t_wall_0)
        cpu_s = max(0.0, t_cpu_1 - t_cpu_0)
        lat_sorted = sorted(latencies_us)

        result_q.put(
            {
                "n_steps": n_steps,
                "wall_s": round(wall_s, 6),
                "reader_cpu_s": round(cpu_s, 6),
                "reader_cpu_ms_per_step": round((cpu_s * 1000.0) / n_steps, 4),
                "reader_cpu_util_pct": round((cpu_s / wall_s) * 100.0, 2),
                "sched_yield_total": yield_count,
                "sched_yield_per_step": round(yield_count / n_steps, 2),
                "poller_poll_total": poll_count,
                "poller_poll_per_step": round(poll_count / n_steps, 2),
                "wakeup_latency_us_mean": round(statistics.mean(lat_sorted), 2),
                "wakeup_latency_us_p50": round(_percentile(lat_sorted, 50.0), 2),
                "wakeup_latency_us_p95": round(_percentile(lat_sorted, 95.0), 2),
            }
        )
    finally:
        reader.shutdown()


def run_workload(
    workload_name: str,
    step_interval_s: float,
    n_steps: int,
    busy_loop_s: float,
    n_warmup: int = 10,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    ready_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()

    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=64 * 1024,
        max_chunks=4,
    )
    proc = ctx.Process(
        target=_reader_worker,
        args=(
            writer.export_handle(),
            busy_loop_s,
            n_warmup,
            n_steps,
            ready_q,
            result_q,
        ),
    )
    proc.start()
    try:
        writer.wait_until_ready()
        assert ready_q.get(timeout=30) == "ready"

        payload = list(range(64))
        for i in range(n_warmup):
            time.sleep(0.0005)
            writer.enqueue(("warmup", i, time.monotonic_ns(), payload))

        # Wait briefly for reader to reset counters after warmup
        time.sleep(0.001)

        next_deadline = time.monotonic()
        for i in range(n_steps):
            next_deadline += step_interval_s
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            writer.enqueue(("step", i, time.monotonic_ns(), payload))

        res = result_q.get(timeout=60)
        proc.join(timeout=10)
        assert proc.exitcode == 0, f"Reader exited with {proc.exitcode}"
        return res
    finally:
        writer.shutdown()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    assert shm_broadcast.SHM_SPIN_BUSY_LOOP_S == 0.002, (
        f"Expected SHM_SPIN_BUSY_LOOP_S == 0.002, got {shm_broadcast.SHM_SPIN_BUSY_LOOP_S}"
    )

    workloads = [
        ("decode_step_21ms", 0.021, 200),
        ("prefill_chunk_250ms", 0.250, 20),
        ("sub_ms_rpc_burst_0p2ms", 0.0002, 500),
    ]

    rows: list[dict[str, Any]] = []
    for name, interval_s, n_steps in workloads:
        base_res = run_workload(name, interval_s, n_steps, busy_loop_s=1.0)
        cand_res = run_workload(
            name, interval_s, n_steps, busy_loop_s=shm_broadcast.SHM_SPIN_BUSY_LOOP_S
        )
        cpu_reduction_pct = (
            (1.0 - (cand_res["reader_cpu_s"] / base_res["reader_cpu_s"])) * 100.0
            if base_res["reader_cpu_s"] > 0
            else 0.0
        )
        row = {
            "boot": args.boot,
            "workload": name,
            "step_interval_ms": round(interval_s * 1000.0, 3),
            "n_steps": n_steps,
            "baseline_busy_loop_s": 1.0,
            "candidate_busy_loop_s": shm_broadcast.SHM_SPIN_BUSY_LOOP_S,
            "baseline": base_res,
            "candidate": cand_res,
            "reader_cpu_reduction_pct": round(cpu_reduction_pct, 2),
        }
        rows.append(row)
        print(json.dumps(row))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
