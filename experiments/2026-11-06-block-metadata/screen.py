#!/usr/bin/env python3
"""CPU work-accounting screen; deliberately not a serving benchmark.

Exercise the pinned BlockTables.append_block_ids and StagedWriteTensor.stage_write
methods without importing CUDA. Device application/gather are Python reference
operations. The geometry is one 256-token paged group and one circular group;
bytes scale by the actual group's kernel-block expansion and group count.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def method(path, class_name, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fun = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fun], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)
    return namespace[name]


class Counts:
    def __init__(self, groups, slots):
        self.values = [[0] * slots for _ in range(groups)]

    def __getitem__(self, key):
        group, slot = key
        return self.values[group][slot]

    def __setitem__(self, key, value):
        group, slot = key
        self.values[group][slot] = value


def spans(old, new, gap):
    """Exact spans, optionally transmitting unchanged gaps to reduce descriptors."""
    result = []
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b:
            if result and i - result[-1][1] <= gap:
                result[-1][1] = i + 1
            else:
                result.append([i, i + 1])
    return result


def run(source):
    bt_path = source / "vllm/v1/worker/gpu/block_table.py"
    sw_path = source / "vllm/v1/worker/gpu/buffer_utils.py"
    append = method(bt_path, "BlockTables", "append_block_ids")
    stage = method(sw_path, "StagedWriteTensor", "stage_write")
    records = []
    comparisons = 0
    for context in [32768, 65536, 262144, 524288, 1048576]:
        for concurrency in [1, 8, 16]:
            # The configured table is always a 1M table. Leave room for a page
            # crossing that reaches exactly the visible 1M limit in the last test.
            width = 4096
            live = min(context // 256, width - 1)
            widths = [width, 1]
            tables = []
            for w in widths:
                t = SimpleNamespace(gpu=SimpleNamespace(shape=(concurrency, w)),
                                    _staged_write_indices=[], _staged_write_starts=[],
                                    _staged_write_contents=[], _staged_write_cu_lens=[])
                t.stage_write = lambda row, start, x, t=t: stage(t, row, start, x)
                tables.append(t)
            manager = SimpleNamespace(num_kv_cache_groups=2, num_blocks=SimpleNamespace(np=Counts(2, concurrency)),
                                      blocks_per_kv_block=[1, 1], block_tables=tables)
            source_rows = [[[0] * w for _ in range(concurrency)] for w in widths]
            previous = [[[0] * w for _ in range(concurrency)] for w in widths]
            mapping = list(range(concurrency))
            # Fragment physical IDs without fragmenting logical append ranges.
            expected = [[[(r + 1) * 100000 + 17 * i + 1 for i in range(live)]
                         for r in range(concurrency)], [[900000 + r] for r in range(concurrency)]]

            def emit(name, updates, selected=None):
                nonlocal previous, comparisons
                for slot, ids, overwrite in updates:
                    append(manager, slot, ids, overwrite)
                staged_words = sum(len(t._staged_write_contents) for t in tables)
                staged_spans = sum(len(t._staged_write_indices) for t in tables)
                for g, t in enumerate(tables):
                    start_content = 0
                    for row, first, end_content in zip(t._staged_write_indices, t._staged_write_starts, t._staged_write_cu_lens):
                        values = t._staged_write_contents[start_content:end_content]
                        source_rows[g][row][first:first + len(values)] = values
                        start_content = end_content
                    for field in ["indices", "starts", "contents", "cu_lens"]:
                        getattr(t, "_staged_write_" + field).clear()
                active = mapping if selected is None else selected
                current = []
                for g, w in enumerate(widths):
                    group = []
                    for row in range(concurrency):
                        if row < len(active):
                            slot = active[row]
                            count = manager.num_blocks.np[g, slot]
                            value = source_rows[g][slot][:count] + [0] * (w - count)
                            oracle = expected[g][slot] + [0] * (w - len(expected[g][slot]))
                            assert value == oracle, (name, g, row)
                        else:
                            value = [0] * w
                        group.append(value)
                    current.append(group)
                modes = {}
                for label, gap in [("changed_span", 0), ("coalesced_16_words", 16), ("coalesced_64_words", 64)]:
                    words = descriptors = 0
                    for a_group, b_group in zip(previous, current):
                        for a, b in zip(a_group, b_group):
                            changes = spans(a, b, gap)
                            reconstructed = a.copy()
                            for lo, hi in changes:
                                reconstructed[lo:hi] = b[lo:hi]
                                words += hi - lo
                            descriptors += len(changes)
                            assert reconstructed == b
                            comparisons += 1
                    modes[label] = {"written_bytes": 4 * words, "descriptors": descriptors,
                                    "host_compared_words": concurrency * sum(widths)}
                records.append({"context_label": context, "live_tokens_before_growth": live * 256,
                                "concurrency": concurrency, "event": name,
                                "producer_delta": {"payload_bytes": 4 * staged_words, "spans": staged_spans,
                                                   "host_compared_words": 0},
                                "full_gather": {"written_bytes": 4 * concurrency * sum(widths),
                                                "gpu_launches": 1}, "snapshot_alternatives": modes})
                previous = current

            emit("new_requests", [(r, (expected[0][r], expected[1][r]), True) for r in mapping])
            emit("growth_inside_page", [])
            for r in mapping:
                expected[0][r].append(700000 + r)
            emit("page_boundary", [(r, ([700000 + r], []), False) for r in mapping])
            emit("circular_window_wrap", [])
            mapping.reverse()
            emit("compacted_batch_reorder", [])
            # Sparse logical changes require an overwrite, not an append. This
            # is API stress, not a claim that the active scheduler emits it.
            for r in mapping:
                for i in range(0, len(expected[0][r]), 17):
                    expected[0][r][i] = 0
            emit("fragmented_overwrite", [(r, (expected[0][r], expected[1][r]), True) for r in mapping])
            emit("preempted_batch", [], selected=[])
            for r in mapping:
                expected[0][r] = [800000 + r * 10000 + i for i in range(live)]
            emit("resume_replaced_pages", [(r, (expected[0][r], expected[1][r]), True) for r in mapping])
            # Same slot and count, different request/pages: counts alone cannot
            # validate an incremental destination cache.
            for r in mapping:
                expected[0][r] = [x + 2000000 for x in expected[0][r]]
                expected[1][r] = [3000000 + r]
            emit("same_size_slot_reuse", [(r, (expected[0][r], expected[1][r]), True) for r in mapping])
            for r in mapping:
                expected[0][r] = [4000000 + r]
                expected[1][r] = [5000000 + r]
            emit("shorter_slot_reuse_tail_zero", [(r, (expected[0][r], expected[1][r]), True) for r in mapping])
    return {"kind": "CPU work accounting and staging semantics; no GPU timing or serving proof",
            "geometry": {"paged_block_tokens": 256, "configured_max_tokens": 1048576,
                         "group_widths": [4096, 1], "note": "two-group fixture, not a measured runtime cache layout"},
            "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [bt_path, sw_path]},
            "cases": len(records), "exact_reference_comparisons": comparisons, "failures": 0, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.source), indent=2) + "\n")
