#!/usr/bin/env python3
"""Benchmark NewRequestData MessageQueue IPC and StagedWriteTensor staging."""

from __future__ import annotations

import json
import multiprocessing as mp
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


@dataclass
class StockNewRequestData:
    """Stock NewRequestData behavior: two distinct list[int] objects."""

    req_id: str
    prompt_token_ids: list[int]
    prefill_token_ids: list[int]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


def _reader_worker(handle, rank: int, num_iters: int, result_q: mp.Queue) -> None:
    mq = MessageQueue.create_from_handle(handle, rank=rank)
    mq.wait_until_ready()
    # Warmup
    _ = mq.dequeue(timeout=10.0)

    dequeue_ms: list[float] = []
    np_convert_ms: list[float] = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        item = mq.dequeue(timeout=10.0)
        t1 = time.perf_counter()
        arr = np.asarray(item.prefill_token_ids, dtype=np.int32)
        t2 = time.perf_counter()
        assert arr.shape[0] == len(item.prompt_token_ids)
        dequeue_ms.append((t1 - t0) * 1000.0)
        np_convert_ms.append((t2 - t1) * 1000.0)

    mq.shutdown()
    result_q.put(
        {
            "rank": rank,
            "dequeue_median_ms": statistics.median(dequeue_ms),
            "np_convert_median_ms": statistics.median(np_convert_ms),
        }
    )


def bench_prompt_len(prompt_len: int, n_readers: int = 4, num_iters: int = 12) -> dict:
    tokens = [(i * 31 + 7) % 128000 for i in range(prompt_len)]
    block_ids = (list(range(prompt_len // 64)),)

    results = {"prompt_len": prompt_len, "n_readers": n_readers, "num_iters": num_iters}
    ctx = mp.get_context("spawn")

    for mode in ("stock", "candidate"):
        mq_writer = MessageQueue(
            n_reader=n_readers,
            n_local_reader=n_readers,
            local_reader_ranks=list(range(n_readers)),
            max_chunk_bytes=16 * 1024 * 1024,
            max_chunks=4,
        )
        handle = mq_writer.export_handle()
        result_q = ctx.Queue()
        procs = [
            ctx.Process(
                target=_reader_worker, args=(handle, r, num_iters, result_q)
            )
            for r in range(n_readers)
        ]
        for p in procs:
            p.start()
        mq_writer.wait_until_ready()

        if mode == "stock":
            payload = StockNewRequestData(
                req_id=f"req-{prompt_len}",
                prompt_token_ids=tokens,
                prefill_token_ids=tokens.copy(),
                block_ids=block_ids,
                num_computed_tokens=0,
            )
        else:
            req = Request(
                request_id=f"req-{prompt_len}",
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(max_tokens=16),
                pooling_params=None,
            )
            payload = NewRequestData.from_request(
                req, block_ids=block_ids, prefill_token_ids=req._all_token_ids
            )

        # Warmup
        mq_writer.enqueue(payload, timeout=10.0)

        enqueue_ms: list[float] = []
        for _ in range(num_iters):
            if mode == "candidate":
                payload = NewRequestData.from_request(
                    req, block_ids=block_ids, prefill_token_ids=req._all_token_ids
                )
            else:
                payload = StockNewRequestData(
                    req_id=f"req-{prompt_len}",
                    prompt_token_ids=tokens,
                    prefill_token_ids=tokens.copy(),
                    block_ids=block_ids,
                    num_computed_tokens=0,
                )
            t0 = time.perf_counter()
            mq_writer.enqueue(payload, timeout=10.0)
            t1 = time.perf_counter()
            enqueue_ms.append((t1 - t0) * 1000.0)

        reader_stats = [result_q.get(timeout=30.0) for _ in range(n_readers)]
        for p in procs:
            p.join(timeout=10.0)
        mq_writer.shutdown()

        # Also measure StagedWriteTensor stage_write + apply_write on GPU/UVA
        device = torch.device("cuda:0")
        staged = StagedWriteTensor(
            (2, prompt_len + 64),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        token_seq = payload.prefill_token_ids
        staged.stage_write(0, 0, token_seq)
        staged.apply_write()
        torch.cuda.synchronize()

        stage_ms: list[float] = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            staged.stage_write(0, 0, token_seq)
            staged.apply_write()
            t1 = time.perf_counter()
            stage_ms.append((t1 - t0) * 1000.0)
        torch.cuda.synchronize()

        med_enq = statistics.median(enqueue_ms)
        med_deq = statistics.median([r["dequeue_median_ms"] for r in reader_stats])
        med_np = statistics.median([r["np_convert_median_ms"] for r in reader_stats])
        med_stg = statistics.median(stage_ms)
        results[mode] = {
            "enqueue_median_ms": round(med_enq, 4),
            "reader_dequeue_median_ms": round(med_deq, 4),
            "reader_np_asarray_median_ms": round(med_np, 4),
            "staged_write_host_median_ms": round(med_stg, 4),
            "total_critical_host_ms": round(med_enq + med_deq + med_np + med_stg, 4),
        }

    stock_tot = results["stock"]["total_critical_host_ms"]
    cand_tot = results["candidate"]["total_critical_host_ms"]
    results["speedup_x"] = round(stock_tot / max(cand_tot, 1e-6), 2)
    return results


def main() -> None:
    for prompt_len in (4096, 32768, 131072, 524288, 1048576):
        row = bench_prompt_len(prompt_len)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
