#!/usr/bin/env python3
"""Standalone CSAN controls and visibility check for candidate-37 copy_alloc.

Run each mode in a fresh isolated GPU process. This launches real GPU work when
invoked; --help only parses arguments. It never starts a server or touches a
cluster. No stream busy-wait, runtime stub or forced launch blocking is used.

Expected eager controls: torch-safe exits 0; torch-race exits 86 with a CSAN
diagnostic. Exit 77 means the writer was not observed by CSAN, NOT a clean race
check. --execution graph tests graph-replay visibility separately. Every run
prints CSAN-RESULT JSON; retain stdout/stderr and exit status together.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE_COMMIT = "5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575"
COPY_SHA256 = "89cd4cf39ad75b8992504afd5bc7d91a088dda0c64a66d314126553a4379187d"


def access_record(access):
    if access is None:
        return None
    return {
        "type": access.type.name,
        "sequence": access.seq_num,
        "stream": access.stream,
        "operator": str(access.operator),
        "aliases": list(access.aliases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("torch-safe", "torch-race", "tilelang-safe", "tilelang-race"))
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    args = parser.parse_args()
    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        parser.error("CUDA_LAUNCH_BLOCKING changes the ordering under investigation")

    import torch
    import torch.cuda._sanitizer as csan

    # Enable before allocations/streams, without requiring an environment change.
    if not csan.cuda_sanitizer.enabled:
        csan.enable_cuda_sanitizer()
    trace = csan.cuda_sanitizer.dispatch.event_handler.tensors_accessed
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    limits = torch.full((4,), 16, dtype=torch.int32, device="cuda")
    host = torch.empty((4,), dtype=torch.int32, device="cpu", pin_memory=True)
    ptr = limits.data_ptr()
    source = None
    keep_alive = []

    if args.mode.startswith("torch"):
        def producer():
            limits.fill_(5)
        expected_final = [5, 5, 5, 5]
    else:
        from vllm.v1.worker.gpu.spec_decode import tilelang_copy as tlc

        source_path = Path(tlc.__file__)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_hash != COPY_SHA256:
            raise RuntimeError(f"copy_alloc source is not candidate-37: {source_path} {source_hash}")
        source = {"path": str(source_path), "sha256": source_hash,
                  "expected_commit": SOURCE_COMMIT}
        idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        total = torch.full((4,), 64, dtype=torch.int32, device="cuda")
        sources = torch.full((1, tlc.CHUNKS, tlc.num_levels(8)), -1,
                             dtype=torch.int32, device="cuda")
        sampled = torch.ones(1, dtype=torch.int32, device="cuda")
        verified = torch.zeros(1, dtype=torch.int32, device="cuda")
        state = torch.zeros((4, tlc.STATE_COLUMNS), dtype=torch.int32, device="cuda")
        alloc = torch.zeros(1, dtype=torch.int32, device="cuda")
        keep_alive = [idx, total, sources, sampled, verified, state, alloc]

        def producer():
            tlc.copy_alloc(idx, total, sources, sampled, verified, verified,
                           state, alloc, limits, 8, 16, 5, 8)
        expected_final = [5, 16, 16, 16]

    # Compile/warm separately. No conflict exists during setup.
    producer()
    torch.cuda.synchronize()
    graph = None
    if args.execution == "graph":
        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(main_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_stream):
            producer()
        torch.cuda.synchronize()
    limits.fill_(16)
    torch.cuda.synchronize()

    before_write = access_record(trace.get_write(ptr))
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(main_stream)
        host.copy_(limits, non_blocking=True)
    copy_reads = [access_record(access) for access in trace.get_reads(ptr)]
    copy_visible = any(access["stream"] == copy_stream.cuda_stream for access in copy_reads)
    ordered = args.mode.endswith("safe")
    if ordered:
        main_stream.wait_stream(copy_stream)

    caught = None
    try:
        if graph is None:
            producer()
        else:
            graph.replay()
    except csan.CUDASanitizerErrors as exc:
        caught = exc
    after_write = access_record(trace.get_write(ptr))
    writer_visible = after_write is not None and after_write != before_write

    # The test's conflicting interval is over. Observe results only now.
    torch.cuda.synchronize()
    final = limits.cpu().tolist()
    copied = host.tolist()
    errors = [] if caught is None else [
        {"pointer": error.data_ptr,
         "current": access_record(error.current_access),
         "previous": access_record(error.previous_access)}
        for error in caught.errors
    ]
    result = {
        "mode": args.mode, "execution": args.execution,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "csan_source": str(Path(csan.__file__)),
        "csan_source_sha256": hashlib.sha256(Path(csan.__file__).read_bytes()).hexdigest(),
        "copy_alloc_source": source,
        "limits_pointer": ptr,
        "main_stream": main_stream.cuda_stream, "copy_stream": copy_stream.cuda_stream,
        "reader_to_next_writer_wait": ordered,
        "write_before": before_write, "write_after": after_write,
        "copy_reads": copy_reads, "copy_read_observed": copy_visible,
        "producer_write_observed": writer_visible,
        "errors": errors, "actual_final_limits": final, "copied_limits": copied,
        "expected_final_limits": expected_final,
        "interpretation": "Visibility/order diagnostic only; no observed overlap or serving NaN is implied.",
    }
    if final != expected_final or (ordered and copied != [16] * 4):
        result["verdict"] = "unexpected kernel/copy values"
        code = 1
    elif not copy_visible:
        result["verdict"] = "coverage gap: D2H source read not observed"
        code = 77
    elif caught is not None:
        expected_race = (not ordered and writer_visible and any(
            error["pointer"] == ptr and error["current"]["type"] == "WRITE"
            and error["previous"]["type"] == "READ" for error in errors
        ))
        result["verdict"] = "expected unsynchronized access detected" if expected_race else "unexpected CSAN report"
        code = 86 if expected_race else 1
    elif not writer_visible:
        result["verdict"] = "coverage gap: actual producer write not observed"
        code = 77
    elif not ordered:
        result["verdict"] = "control failed: visible unordered accesses were not reported"
        code = 1
    else:
        result["verdict"] = "ordered control clean; both accesses observed"
        code = 0
    print("CSAN-RESULT " + json.dumps(result, sort_keys=True), flush=True)
    # Keep references alive through all completion and result reads.
    _ = keep_alive, graph
    return code


if __name__ == "__main__":
    raise SystemExit(main())
