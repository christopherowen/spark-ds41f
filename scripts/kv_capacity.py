#!/usr/bin/env python3
"""How many full-length requests DeepSeek V4.1 Flash's KV pool holds at once, per shape.

usage: kv_capacity.py [--cluster-config FILE] [--model-config FILE] [--shape NAME]...
                      [--max-num-batched-tokens N] [--json]

vLLM's startup line "GPU KV cache size: N tokens, Maximum concurrency for L tokens per
request: Cx" is C x L, with C the requests at the per-request limit L that the pool holds.
Up to spark-ds41f-1025 its vLLM charged every request the window caches' own peak: the
128-token window plus a whole batch for each of the two steps async scheduling keeps
unsettled. Those two batches are the whole pool's, not every request's. Per request the
pool holds:

- the compressed caches of the KV source layers (latent and indexer key): one block per
  ``--block-size`` tokens of context; their pages make one pool block;
- the window caches of every attention layer, the drafter's included, packed as many to a
  group as a pool block holds pages: per group, the blocks under the window plus the decode
  rows of both steps and the drafter's lookahead (three blocks);
- one block per ring group for the ratio-2 compressors' open states;

and once, for the one prompt being filled, its chunk in flight and its next chunk (each at
most the long-prefill threshold) on top of the window. "contexts" is the number of
requests at L the pool holds that way, one of them still prefilling; "every sequence" is
the longest context at which all ``max_num_seqs`` fit at once.

The layout follows vLLM's DS4.1 caches (``vllm/models/deepseek_v4_1``) and its
block-outermost packing (``_get_packed_kv_cache_groups``). ``reported`` is the figure the
startup line printed up to spark-ds41f-1025; ``vllm_capacity`` the one it prints with vLLM
0491 (each request its window, the pool the two batches once) after 0492 (a window group
holding a drafter layer retains the drafter's extra token).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

WINDOW_BLOCK = 128  # CacheConfig.DEFAULT_DS41_SWA_BLOCK_SIZE
WINDOW_BYTES = 528  # a window token: the latent, its scales and RoPE part
LATENT_BYTES = 288  # a compressed state's latent
INDEX_BYTES = 68  # a compressed state's indexer key
RING_SLOT_BYTES = 4096  # a compressor slot: 512 FP32 values and 512 FP32 gates
STEPS_IN_FLIGHT = 2  # async scheduling: the step in flight and the one being scheduled


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Geometry:
    block_size: int  # tokens per compressed-cache block (--block-size)
    window: int  # sliding-window tokens
    pool_block_bytes: int
    window_groups: int
    drafter_groups: int  # window groups holding a drafter layer
    ring_groups: int


def ced_decoder_start(text: dict) -> int | None:
    """The first full-resolution KV source after the compressed encoder (vLLM's rule): its
    window caches and every later layer's keep no prefix-cache entries."""
    ratios = text["compress_ratios"]
    layers = text["num_hidden_layers"]
    for source in sorted(text["kv_source_layer_ids"]):
        if 0 < source < min(layers, len(ratios)) and ratios[source] == 1 and any(r > 1 for r in ratios[:source]):
            return source
    return None


def geometry(text: dict, block_size: int, drafts: int) -> Geometry:
    """The cache groups vLLM packs for the model at ``--block-size`` and ``drafts``."""
    ratios = text["compress_ratios"]
    sources = [s for s in text["kv_source_layer_ids"] if ratios[s]]
    indexed = set(text["index_source_layer_ids"])
    compressed = sum(block_size // ratios[s] * (LATENT_BYTES + INDEX_BYTES * (s in indexed)) for s in sources)
    window_page = WINDOW_BLOCK * WINDOW_BYTES
    ring_page = max(8, 1 << (drafts + 1).bit_length()) * RING_SLOT_BYTES
    rings = sum(ratios[s] == 2 for s in sources)
    # The pool block is the widest group: the compressed one, which stays whole.
    pool_block = max(compressed, window_page, ring_page)
    # Window caches split by prefix-cache policy: layers before the CED decoder keep
    # entries; the decoder's and the drafter's replay a bounded window instead.
    layers = text["num_hidden_layers"]
    boundary = ced_decoder_start(text)
    drafter = text.get("n_mtp_layers") or text.get("num_nextn_predict_layers") or 3  # DSpark's
    buckets = [layers, drafter] if boundary is None else [boundary, layers - boundary + drafter]
    per_block = pool_block // window_page
    # Packing deals a bucket's layers round-robin over its groups, the drafter's last.
    drafter_groups = cdiv(drafter, per_block) if boundary is None else min(drafter, cdiv(buckets[1], per_block))
    return Geometry(
        block_size=block_size,
        window=text["sliding_window"],
        pool_block_bytes=pool_block,
        window_groups=sum(cdiv(n, per_block) for n in buckets if n),
        drafter_groups=drafter_groups,
        ring_groups=cdiv(rings, pool_block // ring_page) if rings else 0,
    )


def capacity(geo: Geometry, sizing: dict, drafts: int) -> dict:
    """The pool's figures at one shape's sizing."""
    length = int(sizing["max_model_len"])
    budget = int(sizing["max_num_batched_tokens"])
    seqs = int(sizing["max_num_seqs"])
    threshold = budget - seqs * (1 + drafts)  # recipe.values' long-prefill threshold
    blocks = int(sizing["kv_cache_memory_bytes"]) // geo.pool_block_bytes
    usable = blocks - 1  # vLLM's null block
    full = cdiv(length, geo.block_size)
    kept = geo.window - 1
    # What the startup line printed: every request at one request's own peak.
    peak = cdiv(min(kept + STEPS_IN_FLIGHT * budget, length), WINDOW_BLOCK) + 1
    reported = blocks / (full + geo.window_groups * peak + geo.ring_groups)
    # A decoding request: the window, the drafter's retained token, both steps' decode
    # rows and the lookahead. The prompt being filled: its two chunks instead of the rows.
    decode_span = kept + 1 + STEPS_IN_FLIGHT * (1 + drafts) + drafts
    resident_window = cdiv(min(decode_span, length), WINDOW_BLOCK) + 1
    resident = full + geo.window_groups * resident_window + geo.ring_groups
    prompt_span = kept + 1 + STEPS_IN_FLIGHT * threshold + drafts
    transient = geo.window_groups * (cdiv(min(prompt_span, length), WINDOW_BLOCK) + 1 - resident_window)
    contexts = (usable - transient) / resident
    each = ((usable - transient) // seqs - (resident - full)) * geo.block_size
    # vLLM kv-capacity: each request its window ((n - 2) / block + 2 blocks for n tokens),
    # the pool both batches once, and never below the per-request-peak figure.
    window_shares = (geo.window_groups * ((kept + drafts - 2) / WINDOW_BLOCK + 2)
                     + geo.drafter_groups / WINDOW_BLOCK)
    shared = geo.window_groups * STEPS_IN_FLIGHT * budget / WINDOW_BLOCK
    vllm_capacity = max(reported, (blocks - shared) / (full + window_shares + geo.ring_groups))
    return {
        "max_model_len": length,
        "max_num_batched_tokens": budget,
        "max_num_seqs": seqs,
        "pool_blocks": blocks,
        "reported_tokens": int(reported * length),
        "reported_contexts": round(reported, 4),
        "resident_blocks": resident,
        "prompt_blocks_in_flight": transient,
        "contexts": round(contexts, 4),
        "full_contexts": int(contexts),
        "every_sequence_tokens": min(length, each),
        "vllm_capacity_tokens": int(vllm_capacity * length),
        "vllm_capacity_contexts": round(vllm_capacity, 4),
    }


def shapes(cluster: dict, names: list[str], budget: int | None) -> dict[str, dict]:
    """Each named shape's sizing (all shapes when none are named)."""
    out = {}
    for name in names or list(cluster["shapes"]):
        sizing = dict(recipe.shape_entry(cluster, name)["sizing"])
        if budget is not None:
            sizing["max_num_batched_tokens"] = budget
        out[name] = sizing
    return out


def block_size(cluster: dict) -> int:
    args = cluster["serve_args"]
    return int(args[args.index("--block-size") + 1])


def report(cluster: dict, text: dict, names: list[str], budget: int | None = None) -> dict[str, dict]:
    drafts = int(cluster["drafts"])
    geo = geometry(text, block_size(cluster), drafts)
    return {name: {**capacity(geo, sizing, drafts), "pool_block_bytes": geo.pool_block_bytes,
                   "window_groups": geo.window_groups, "ring_groups": geo.ring_groups}
            for name, sizing in shapes(cluster, names, budget).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cluster-config", default=str(ROOT / "config/cluster.json"),
                    help="a recipe (config/cluster.json or an experiment's)")
    ap.add_argument("--model-config", default=str(ROOT / "state/config.json"))
    ap.add_argument("--shape", action="append", default=[], help="repeatable; default: every shape")
    ap.add_argument("--max-num-batched-tokens", type=int, help="override every shape's prefill budget")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cluster = json.loads(Path(args.cluster_config).read_text())
    if not recipe.is_recipe(cluster):
        raise SystemExit(f"{args.cluster_config} is not a recipe (no shapes)")
    model = json.loads(Path(args.model_config).read_text())
    rows = report(cluster, model.get("text_config", model), args.shape, args.max_num_batched_tokens)
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    for name, row in rows.items():
        print(f"{name}: {row['max_model_len']:,} tokens, budget {row['max_num_batched_tokens']:,}, "
              f"{row['pool_blocks']:,} blocks of {row['pool_block_bytes']:,} bytes")
        print(f"  holds {row['full_contexts']} full contexts at once ({row['contexts']:.2f}); "
              f"all {row['max_num_seqs']} sequences up to {row['every_sequence_tokens']:,} tokens each")
        print(f"  a request at the limit: {row['resident_blocks']:,} blocks; the prompt being filled: "
              f"{row['prompt_blocks_in_flight']:,} more")
        print(f"  vLLM's startup line: {row['reported_tokens']:,} tokens ({row['reported_contexts']:.2f}x) "
              f"up to spark-ds41f-1025; {row['vllm_capacity_tokens']:,} ({row['vllm_capacity_contexts']:.2f}x) "
              f"with vLLM 0491 and 0492")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
