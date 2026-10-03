"""Read-only layout report for the audited DS4.1 source/model combinations."""

import hashlib
import json
import math
from pathlib import Path

import topology


LAYOUT_PATH = "experiments/2026-10-03-transport-profiles/model-layout.json"


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def describe(root: Path, cluster: dict, capacity_bytes: int) -> dict:
    raw = (root / LAYOUT_PATH).read_bytes()
    audit = json.loads(raw)
    labels = cluster["container"]["expected_labels"]
    pair = {key: labels.get(f"local.spark3.{key}.tree") for key in ("vllm", "b12x")}
    if pair not in audit["source_pairs"]:
        raise ValueError("model layout needs a source audit for these vLLM/B12X trees")
    if not any(audit["model_snapshot"] in mount[0] and mount[1] == "/models"
               for mount in cluster["container"]["mounts"]):
        raise ValueError("model layout audit does not match the mounted checkpoint")
    tp = int(topology.argument(cluster, "--tensor-parallel-size"))
    dims, alignment = audit["dimensions"], audit["alignment"]
    if tp not in (3, 4) or topology.argument(cluster, "--dtype") != "bfloat16":
        raise ValueError("model layout report covers the audited TP3/TP4 BF16 profiles")

    def sharded(logical, padded):
        return {"logical_global": logical, "padded_global": padded,
                "extra_global": padded - logical, "allocated_per_rank": padded // tp}

    groups = round_up(dims["attention_output_groups"], tp)
    heads = groups * (dims["attention_heads"] // dims["attention_output_groups"])
    vocab = round_up(dims["vocabulary_rows"], math.lcm(alignment["global_vocabulary_rows"], tp))
    expert = dims["expert_intermediate_width"] // tp
    compile_config = json.loads(topology.argument(cluster, "--compilation-config"))
    graph_sizes = compile_config["cudagraph_capture_sizes"]
    draft = json.loads(topology.argument(cluster, "--speculative-config"))
    max_decode = max(int(topology.argument(cluster, "--max-cudagraph-capture-size")),
                     int(topology.argument(cluster, "--max-num-seqs")) * (1 + draft["num_speculative_tokens"]), tp)
    # sp_prefill.rows_above_bytes: the fewest live rows whose padded BF16
    # hidden-state buffer exceeds registered capacity, independent of dispatch.
    sp_min = max(max_decode + 1, round_up(capacity_bytes // (dims["hidden_size"] * 2) + 1, tp) - tp + 1)
    examples = []
    for rows in (1, 6, 19, 48, 205, 4096):
        sp_rows = round_up(rows, tp) if rows >= sp_min else None
        examples.append({"live_rows": rows,
                         "next_decode_graph_capacity_if_eligible": next((n for n in graph_sizes if n >= rows), None),
                         "prefill_sp_collective_rows_if_eligible": sp_rows,
                         "prefill_sp_extra_rows": sp_rows - rows if sp_rows is not None else None})
    return {
        "authority": "derived audit, not runtime options or a live observation",
        "audit_file": LAYOUT_PATH,
        "audit_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence": audit["evidence"],
        "model_snapshot": audit["model_snapshot"],
        "source_trees": pair,
        "tensor_parallel_size": tp,
        "model_dimensions": {
            "attention_heads": sharded(dims["attention_heads"], heads),
            "attention_output_groups": sharded(dims["attention_output_groups"], groups),
            "engram_wkv_width": sharded(dims["engram_wkv_width"], round_up(dims["engram_wkv_width"], tp * alignment["engram_rows_per_rank"])),
            "target_vocabulary_rows": sharded(dims["vocabulary_rows"], vocab),
            "routed_expert_intermediate_width": sharded(dims["expert_intermediate_width"], dims["expert_intermediate_width"]),
        },
        "kernel_scratch": {
            "compact_moe_n64_path": expert % 128 == 64,
            "compact_moe_intermediate_width_when_selected": round_up(expert, alignment["compact_moe_intermediate_channels"]) if expert % 128 == 64 else None,
            "note": "Scratch alignment is not expert weight padding; actual kernel selection also depends on routed rows.",
        },
        "scheduled_rows": {
            "target_graph_capacities": graph_sizes,
            "prefill_sp_min_live_rows": sp_min,
            "prefill_chunk_limit": int(topology.argument(cluster, "--max-num-batched-tokens")),
            "examples": examples,
            "note": "Rows are per forward, not full prompt length. Graph and SP columns describe separate eligible paths; null means outside this path's range. SP applies to eligible encoder work before CED compaction.",
        },
        "transport_layout": {
            "pack_bytes": alignment["rocenante_pack_bytes"],
            "all_gather": "Aligned rows use direct layout; other supported shapes use padded contiguous scratch rounded to 16 bytes, then reshape. Eligibility still applies.",
        },
        "other": {
            "vision_weights": "replicated at TP3 and TP4 in the served SM121 implementation",
            "drafter": "Uses the same vocabulary partition rule; NVFP4 packing and auxiliary graph workspaces are separate and not fully enumerated by this report.",
            "determinism": "These profiles do not enable experimental batch-invariant LM-head row duplication.",
        },
    }
