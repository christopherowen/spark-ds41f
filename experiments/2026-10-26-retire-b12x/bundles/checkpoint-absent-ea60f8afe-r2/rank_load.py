#!/usr/bin/env python3
"""One TP rank's share of the checkpoint read into CUDA tensors through a loader's weights
iterator and session, then the weight digest (a kernel-lab script, run in the image).

usage: rank_load.py CHECKPOINT --loader direct|b12x --tp 4 --rank 0 --out JSON [--sample-mib 256]

- `direct` is vLLM's DirectModelLoader with its DirectWeightSession (0499, mounted);
- `b12x` is B12X's own B12xModelLoader with B12X's DirectWeightSession in bounce mode, the
  loader production serves with (VLLM_PLUGINS=b12x_loader).

Both run the same shard plan, a stand-in for a rank's (the model is not built: it needs
every rank):
- rank r's 1/tp of every 2-D tensor whose dimensions both divide, by rows or by columns
  (a strided copy; every other name, by CRC-32);
- the whole of every other tensor;
- BF16 vectors expanded to FP32 (in the ring);
- FP32 tensors converted to BF16 (the GPU scratch);
- scalars and declared metadata as CPU values (host copies);
- DS4.1's Engram tables (its checkpoint_file_weight_filter) as file-source descriptors,
  never read.

Every copy goes through weight_transfer.copy_weight, as model weight loaders call it, and
the transfers finish as DefaultModelLoader.load_weights finishes them. Then
weight_digest.weight_digest (0498) hashes the loaded tensors in name order, and each
tensor's SHA-256 is recorded. /proc/meminfo is sampled every 0.1 s from before the session
to the end of the reads. A sample of small tensors is compared afterwards with
safetensors' own reads of the same slices; it runs last, so its page-cache reads are
outside the memory figures. Prints one JSON line; the per-tensor digests and the
descriptors (offset and shape) go to --out.
"""

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import regex as re
import torch

ENGRAM = re.compile(r"layers\.\d+\.engram\.embed\.(?:weight|scale)")


def meminfo() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemAvailable", "MemFree", "Cached"):
            values[key] = int(rest.split()[0]) * 1024
    return values


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.first = meminfo()
        self.low_available = self.first["MemAvailable"]
        self.low_free = self.first["MemFree"]
        self.high_cached = self.first["Cached"]
        self.samples = 1
        self.done = threading.Event()

    def run(self):
        while not self.done.wait(0.1):
            sample = meminfo()
            self.low_available = min(self.low_available, sample["MemAvailable"])
            self.low_free = min(self.low_free, sample["MemFree"])
            self.high_cached = max(self.high_cached, sample["Cached"])
            self.samples += 1

    def stop(self) -> dict:
        self.done.set()
        self.join()
        last = meminfo()
        gib = 1 << 30
        return {
            "samples": self.samples,
            "available_before_gib": round(self.first["MemAvailable"] / gib, 3),
            "available_low_gib": round(min(self.low_available, last["MemAvailable"]) / gib, 3),
            "available_after_gib": round(last["MemAvailable"] / gib, 3),
            "free_before_gib": round(self.first["MemFree"] / gib, 3),
            "free_low_gib": round(min(self.low_free, last["MemFree"]) / gib, 3),
            "cached_before_gib": round(self.first["Cached"] / gib, 3),
            "cached_high_gib": round(max(self.high_cached, last["Cached"]) / gib, 3),
            "cached_after_gib": round(last["Cached"] / gib, 3),
        }


def plan(name: str, tensor: torch.Tensor, tp: int, rank: int):
    """The view this rank loads and the destination dtype."""
    dtype = tensor.dtype
    if tensor.dim() == 1 and dtype == torch.bfloat16:
        dtype = torch.float32
    elif dtype == torch.float32 and tensor.dim() >= 1:
        dtype = torch.bfloat16
    if tensor.dim() == 2 and tensor.shape[0] % tp == 0 and tensor.shape[1] % tp == 0:
        if dtype == tensor.dtype and zlib.crc32(name.encode()) % 2:
            width = tensor.shape[1] // tp
            return tensor[:, rank * width : (rank + 1) * width], dtype
        height = tensor.shape[0] // tp
        return tensor[rank * height : (rank + 1) * height], dtype
    return tensor, dtype


def sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1).view(torch.uint8)
    step = 64 << 20
    for start in range(0, flat.numel(), step):
        digest.update(flat[start : start + step].cpu().numpy())
    return digest.hexdigest()


def loader_and_session(kind: str, device: int):
    from vllm.config.load import LoadConfig

    if kind == "direct":
        from vllm.model_executor.model_loader.direct_io import DirectWeightSession
        from vllm.model_executor.model_loader.direct_loader import DirectModelLoader

        loader = DirectModelLoader(LoadConfig(load_format="direct"))
        return loader, DirectWeightSession(device, io_threads=loader.io_threads)
    from b12x.integration.vllm.loader import B12xModelLoader
    from b12x.loader._checkpoint import DirectWeightSession

    loader = B12xModelLoader(
        LoadConfig(load_format="b12x", model_loader_extra_config={"read_mode": "bounce"})
    )
    return loader, DirectWeightSession(
        device, io_threads=loader.io_threads, read_mode="bounce"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--loader", choices=("direct", "b12x"), required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-mib", type=int, default=256)
    args = parser.parse_args()

    from vllm.model_executor.model_loader import weight_digest
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.weight_transfer import (
        copy_weight,
        finish_weight_transfers,
        get_file_tensor_source,
        weight_transfer,
    )

    device = torch.accelerator.current_device_index()
    torch.cuda.init()
    torch.empty(1, device="cuda")
    sampler = Sampler()
    sampler.start()
    started = time.perf_counter()
    loader, session = loader_and_session(args.loader, device)
    source = DefaultModelLoader.Source(
        args.checkpoint,
        revision=None,
        file_weight_filter=lambda name: ENGRAM.fullmatch(name) is not None,
    )
    loaded, descriptors, values = {}, {}, 0
    with session, weight_transfer(session):
        loader._session = session
        for name, tensor in loader._get_weights_iterator(source):
            file_source = get_file_tensor_source(tensor)
            if file_source is not None:
                descriptors[name] = [file_source.offset, list(file_source.shape)]
                continue
            values += tensor.device.type == "cpu"
            view, dtype = plan(name, tensor, args.tp, args.rank)
            destination = torch.empty(view.shape, dtype=dtype, device="cuda")
            copy_weight(destination, view)
            loaded[name] = destination
        finish_weight_transfers()
        torch.accelerator.synchronize()
        load_seconds = time.perf_counter() - started
        stats = session.stats()
        loader._session = None
    reserved = torch.accelerator.memory_reserved()
    memory = sampler.stop()

    digested = time.perf_counter()
    model = SimpleNamespace(named_parameters=lambda: iter(loaded.items()))
    digest, count, nbytes = weight_digest.weight_digest(model)
    digest_seconds = time.perf_counter() - digested
    with ThreadPoolExecutor(8) as pool:
        per_tensor = dict(zip(loaded, pool.map(sha256, loaded.values()), strict=True))

    sampled = checked = 0
    if args.sample_mib:
        from safetensors import safe_open

        index = json.loads((Path(args.checkpoint) / "model.safetensors.index.json").read_text())
        small = sorted(name for name, t in loaded.items() if t.nbytes <= 16 << 20)
        random.Random(f"{args.tp}/{args.rank}").shuffle(small)
        budget = args.sample_mib << 20
        for name in small:
            if budget <= 0:
                break
            with safe_open(Path(args.checkpoint) / index["weight_map"][name], framework="pt") as archive:
                real = archive.get_tensor(name)
            view, dtype = plan(name, real, args.tp, args.rank)
            expected = view.to(dtype).contiguous()
            budget -= real.nbytes
            sampled += 1
            checked += torch.equal(
                expected.reshape(-1).view(torch.uint8),
                loaded[name].cpu().reshape(-1).view(torch.uint8),
            )

    Path(args.out).write_text(json.dumps({"digest": digest, "tensors": per_tensor, "descriptors": descriptors}) + "\n")
    print(json.dumps({
        "loader": args.loader,
        "tp": args.tp,
        "rank": args.rank,
        "tensors": count,
        "loaded_gib": round(nbytes / (1 << 30), 3),
        "values": values,
        "descriptors": len(descriptors),
        "load_seconds": round(load_seconds, 2),
        "read_gb_per_s": round(stats["payload_bytes"] / 1e9 / load_seconds, 2),
        "digest": digest,
        "digest_seconds": round(digest_seconds, 1),
        "reserved_gib": round(reserved / (1 << 30), 3),
        "memory": memory,
        "loader_overhead_gib": round(
            memory["available_before_gib"] - memory["available_low_gib"] - reserved / (1 << 30), 3
        ),
        "sample": {"tensors": sampled, "equal": checked},
        "stats": stats,
    }), flush=True)
    return 0 if checked == sampled else 1


if __name__ == "__main__":
    sys.exit(main())
