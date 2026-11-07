# SPDX-License-Identifier: Apache-2.0
"""Benchmark 1-image and 4-image ViT + aligner latency and collective volume:
old shard_dim=0 + sequential per-image loop vs new shard_dim=1 row-parallel +
packed multi-image batching."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm.distributed as distributed
from vllm.models.deepseek_v4_1.nvidia import tilelang_vision
from vllm.models.deepseek_v4_1.tilelang import gather as gather_kernels
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang import vision as kernels

DEVICE = "cuda"

FULL_CONFIG = SimpleNamespace(
    vision_patch_size=14,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_n_layers=32,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    hidden_size=5120,
    vision_max_n_token=1024,
)


def _patches(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    patch_dim = 3 * FULL_CONFIG.vision_patch_size**2
    return (
        torch.randn((rows, patch_dim), generator=gen, device=DEVICE, dtype=torch.float32)
        * 0.25
    ).to(torch.bfloat16)


class _OldShard0Runner:
    """Reproduces the pre-rework shard_dim=0 + sequential loop execution and
    kernels (4 all-gathers + gather_columns_into per ViT block + 2 in aligner)."""

    def __init__(self, config, tp_size: int):
        self.config = config
        self.tp_size = tp_size
        self.capacity = tilelang_vision._capacity(config)
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.local_heads = tilelang_vision._local_heads(config, tp_size)
        self.qkv_width = self.local_heads * self.head_dim
        self.local_hidden = tilelang_vision._local_width(config.vision_dim, tp_size)
        self.local_inter = tilelang_vision._local_width(config.vision_inter_dim, tp_size)
        self.out_dim = config.hidden_size
        self.local_out_dim = tilelang_vision._local_width(self.out_dim, tp_size)
        self.downsample_ratio = config.vision_downsample_ratio
        r = self.downsample_ratio
        self.merged_capacity = (self.capacity + r - 1) // r

        patch = 3 * config.vision_patch_size**2
        padded = kernels.padded_width(patch)
        hidden = config.vision_dim
        inter_full = config.vision_inter_dim
        merged_width = hidden * r * r

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device=DEVICE)

        self.patch_w = empty(hidden, patch)
        self.patch_b = empty(hidden)
        self.norm1_w = empty(hidden)
        self.norm2_w = empty(hidden)
        self.wqkv_w = empty(3 * self.qkv_width, hidden)
        self.wqkv_b = empty(3 * self.qkv_width)
        self.wo_w = empty(self.local_hidden, hidden)
        self.wo_b = empty(self.local_hidden)
        self.w1_w = empty(2 * self.local_inter, hidden)
        self.w2_w = empty(self.local_hidden, inter_full)
        self.align_w1_w = empty(self.local_out_dim, merged_width)
        self.align_w1_b = empty(self.local_out_dim)
        self.align_w2_w = empty(self.local_out_dim, self.out_dim)
        self.align_w2_b = empty(self.local_out_dim)

        for t in (
            self.patch_w,
            self.patch_b,
            self.norm1_w,
            self.norm2_w,
            self.wqkv_w,
            self.wqkv_b,
            self.wo_w,
            self.wo_b,
            self.w1_w,
            self.w2_w,
            self.align_w1_w,
            self.align_w1_b,
            self.align_w2_w,
            self.align_w2_b,
        ):
            t.normal_(0, 0.02)

        self.ws_patches = empty(self.capacity, padded)
        self.ws_patch_weight = empty(hidden, padded)
        self.ws_state = empty(self.capacity, hidden)
        self.ws_residual = empty(self.capacity, hidden)
        self.ws_normalized = empty(self.capacity, hidden)
        self.ws_branch = empty(self.capacity, hidden)
        self.ws_qkv = empty(self.capacity, 3 * self.qkv_width)
        self.ws_local_attended = empty(self.capacity, self.qkv_width)
        self.ws_local_branch = empty(self.capacity, self.local_hidden)
        self.ws_activated = empty(self.capacity, inter_full)
        self.ws_local_activated = empty(self.capacity, self.local_inter)
        self.ws_merged = empty(self.merged_capacity, merged_width)
        self.ws_align_hidden = empty(self.merged_capacity, self.out_dim)
        self.ws_align_local = empty(self.merged_capacity, self.local_out_dim)
        inv = 1.0 / (
            config.vision_rope_theta
            ** (
                torch.arange(0, self.head_dim // 2, 2, dtype=torch.float32, device="cpu")
                / (self.head_dim // 2)
            )
        )
        self.inv_freq = inv.to(device=DEVICE)

        calls = {
            "pad": kernels.pad_call(patch),
            "patch": kernels.linear_call(hidden, padded, bias=True),
            "norm": kernels.norm_call(hidden, add=False),
            "qkv": kernels.linear_call(3 * self.qkv_width, hidden, bias=True),
            "rope": kernels.rope_call(self.local_heads, self.head_dim),
            "attention": kernels.attention_call(self.local_heads, self.head_dim),
            "wo": kernels.linear_call(self.local_hidden, hidden, bias=True),
            "add_norm": kernels.norm_call(hidden, add=True),
            "mlp_in": kernels.swiglu_call(self.local_inter, hidden),
            "mlp_out": kernels.linear_call(self.local_hidden, inter_full, bias=False),
            "merge": kernels.merge_call(hidden, r),
            "align_w1": kernels.linear_call(
                self.local_out_dim, merged_width, bias=True, gelu=True
            ),
            "align_w2": kernels.linear_call(
                self.local_out_dim, self.out_dim, bias=True
            ),
        }
        if tp_size > 1:
            for role, loc, full in (
                ("attended", self.qkv_width, hidden),
                ("branch", self.local_hidden, hidden),
                ("activated", self.local_inter, inter_full),
                ("align", self.local_out_dim, self.out_dim),
            ):
                for packed in (False, True):
                    calls[f"gather_{role}_{packed}"] = gather_kernels.gather_call(
                        tp_size, loc, full, 1, torch.bfloat16, packed
                    )
        self.kernels = {k: kernels.resolve(v) for k, v in calls.items()}

    def _gather_cols(self, local: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        gathered = distributed.tensor_model_parallel_all_gather(local, dim=0)
        gather_kernels.gather_columns_into(gathered, (dst,), self.tp_size)
        return dst

    def encode_sequential(
        self, patches: torch.Tensor, vit_grid: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        k = self.kernels
        sharded = self.tp_size > 1
        r = self.downsample_ratio
        outs: list[torch.Tensor] = []
        offset = 0
        for h, w in vit_grid:
            rows = h * w
            item = patches[offset : offset + rows]
            offset += rows
            state = self.ws_state[:rows]
            residual = self.ws_residual[:rows]
            normalized = self.ws_normalized[:rows]
            branch = self.ws_branch[:rows]
            qkv = self.ws_qkv[:rows]
            local_attended = self.ws_local_attended[:rows]
            local_branch = self.ws_local_branch[:rows]
            activated = self.ws_activated[:rows]
            local_activated = self.ws_local_activated[:rows]

            k["pad"](item.reshape(rows, -1), self.ws_patches[:rows])
            k["pad"](self.patch_w, self.ws_patch_weight)
            k["patch"](self.ws_patches[:rows], self.ws_patch_weight, self.patch_b, state)
            k["norm"](state, self.norm1_w, normalized)
            for _ in range(self.config.vision_n_layers):
                k["qkv"](normalized, self.wqkv_w, self.wqkv_b, qkv)
                k["rope"](qkv, self.inv_freq, w)
                k["attention"](qkv, local_attended)
                attended = (
                    self._gather_cols(local_attended, normalized)
                    if sharded
                    else local_attended
                )
                k["wo"](attended, self.wo_w, self.wo_b, local_branch if sharded else branch)
                if sharded:
                    self._gather_cols(local_branch, branch)
                k["add_norm"](state, branch, self.norm2_w, residual, normalized)
                k["mlp_in"](normalized, self.w1_w, local_activated if sharded else activated)
                if sharded:
                    self._gather_cols(local_activated, activated)
                k["mlp_out"](activated, self.w2_w, local_branch if sharded else branch)
                if sharded:
                    self._gather_cols(local_branch, branch)
                k["add_norm"](residual, branch, self.norm1_w, state, normalized)

            m_rows = ((h + r - 1) // r) * ((w + r - 1) // r)
            merged = self.ws_merged[:m_rows]
            align_hidden = self.ws_align_hidden[:m_rows]
            align_local = self.ws_align_local[:m_rows]
            out = torch.empty(
                (m_rows, self.out_dim), dtype=torch.bfloat16, device=DEVICE
            )
            k["merge"](normalized, merged, h, w)
            if sharded:
                k["align_w1"](merged, self.align_w1_w, self.align_w1_b, align_local)
                self._gather_cols(align_local, align_hidden)
                k["align_w2"](align_hidden, self.align_w2_w, self.align_w2_b, align_local)
                self._gather_cols(align_local, out)
            else:
                k["align_w1"](merged, self.align_w1_w, self.align_w1_b, align_hidden)
                k["align_w2"](align_hidden, self.align_w2_w, self.align_w2_b, out)
            outs.append(out)
        return outs


def _build_new_tower(config, tp_size: int):
    with torch.device(DEVICE):
        vit = tilelang_vision.DeepseekV4ViT(config, tp_size=tp_size, tp_rank=0)
        aligner = tilelang_vision.DeepseekV4Aligner(config, tp_size=tp_size, tp_rank=0)
    for mod in (vit, aligner):
        for p in mod.parameters():
            p.data.normal_(0, 0.02)
    vit.declare_kernels()
    aligner.declare_kernels()
    return vit, aligner


def _measure(fn, warmup: int = 5, iters: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(iters):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(float(start_ev.elapsed_time(end_ev)))
    return {
        "median_ms": round(statistics.median(times_ms), 4),
        "mean_ms": round(statistics.mean(times_ms), 4),
        "stdev_ms": round(statistics.stdev(times_ms), 4) if len(times_ms) > 1 else 0.0,
        "min_ms": round(min(times_ms), 4),
        "max_ms": round(max(times_ms), 4),
        "samples_ms": [round(x, 4) for x in times_ms],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    workloads = {
        "1img_24x24_576p": [(24, 24)],
        "4img_24x24_2304p": [(24, 24), (24, 24), (24, 24), (24, 24)],
        "1img_48x48_2304p": [(48, 48)],
        "4img_48x48_9216p": [(48, 48), (48, 48), (48, 48), (48, 48)],
        "1img_96x96_9216p": [(96, 96)],
    }

    results: dict = {
        "boot": args.boot,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "vision_n_layers": FULL_CONFIG.vision_n_layers,
        "vision_capacity_patches": tilelang_vision._capacity(FULL_CONFIG),
        "comparisons": {},
    }

    orig_all_gather = distributed.tensor_model_parallel_all_gather

    for tp_size in (1, 3, 4):
        old_runner = _OldShard0Runner(FULL_CONFIG, tp_size)
        new_vit, new_aligner = _build_new_tower(FULL_CONFIG, tp_size)
        tp_key = f"tp{tp_size}"
        results["comparisons"][tp_key] = {}

        for w_name, grid in workloads.items():
            total_patches = sum(h * w for h, w in grid)
            patches = _patches(total_patches, seed=100 + total_patches)

            stats = {"calls": 0, "local_bytes": 0, "gathered_bytes": 0}
            gather_pool: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = {}

            def mock_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
                stats["calls"] += 1
                stats["local_bytes"] += tensor.nbytes
                stats["gathered_bytes"] += tensor.nbytes * tp_size
                key = (tuple(tensor.shape), tensor.dtype)
                buf = gather_pool.get(key)
                if buf is None:
                    buf = tensor.repeat(tp_size, 1)
                    gather_pool[key] = buf
                return buf

            distributed.tensor_model_parallel_all_gather = mock_all_gather
            try:
                stats = {"calls": 0, "local_bytes": 0, "gathered_bytes": 0}
                old_Runner_fn = lambda: old_runner.encode_sequential(patches, grid)
                old_Runner_fn()
                old_coll = dict(stats)
                old_timing = _measure(old_Runner_fn)

                stats = {"calls": 0, "local_bytes": 0, "gathered_bytes": 0}
                new_seq_fn = lambda: [
                    new_aligner(
                        new_vit(
                            patches[
                                sum(grid[j][0] * grid[j][1] for j in range(i)) : sum(
                                    grid[j][0] * grid[j][1] for j in range(i + 1)
                                )
                            ],
                            h,
                            w,
                        ),
                        h,
                        w,
                    )
                    for i, (h, w) in enumerate(grid)
                ]
                new_seq_fn()
                new_seq_coll = dict(stats)
                new_seq_timing = _measure(new_seq_fn)

                stats = {"calls": 0, "local_bytes": 0, "gathered_bytes": 0}
                new_packed_fn = lambda: tilelang_vision.encode_images(
                    new_vit, new_aligner, patches, grid
                )
                new_packed_fn()
                new_packed_coll = dict(stats)
                new_packed_timing = _measure(new_packed_fn)
            finally:
                distributed.tensor_model_parallel_all_gather = orig_all_gather

            speedup = round(
                old_timing["median_ms"] / new_packed_timing["median_ms"], 4
            )
            results["comparisons"][tp_key][w_name] = {
                "grid": grid,
                "total_patches": total_patches,
                "old_shard_dim0_sequential": {
                    **old_timing,
                    "collectives": old_coll["calls"],
                    "local_sent_bytes_per_rank": old_coll["local_bytes"],
                    "gathered_bytes_per_rank": old_coll["gathered_bytes"],
                },
                "new_shard_dim1_sequential": {
                    **new_seq_timing,
                    "collectives": new_seq_coll["calls"],
                    "local_sent_bytes_per_rank": new_seq_coll["local_bytes"],
                    "gathered_bytes_per_rank": new_seq_coll["gathered_bytes"],
                },
                "new_shard_dim1_packed": {
                    **new_packed_timing,
                    "collectives": new_packed_coll["calls"],
                    "local_sent_bytes_per_rank": new_packed_coll["local_bytes"],
                    "gathered_bytes_per_rank": new_packed_coll["gathered_bytes"],
                },
                "speedup_median_x": speedup,
                "collective_calls_reduction_x": (
                    round(old_coll["calls"] / new_packed_coll["calls"], 2)
                    if new_packed_coll["calls"] > 0
                    else 1.0
                ),
            }
            print(
                f"[Boot {args.boot} {tp_key} {w_name}] "
                f"old={old_timing['median_ms']:.2f}ms ({old_coll['calls']} coll) -> "
                f"new_seq={new_seq_timing['median_ms']:.2f}ms ({new_seq_coll['calls']} coll) -> "
                f"new_packed={new_packed_timing['median_ms']:.2f}ms ({new_packed_coll['calls']} coll) "
                f"[{speedup:.2f}x]"
            )

        del old_runner, new_vit, new_aligner
        torch.cuda.empty_cache()

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote benchmark receipt to {args.out}")


if __name__ == "__main__":
    main()
