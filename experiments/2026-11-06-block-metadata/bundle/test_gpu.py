# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real staging/gather correctness gates; these require an owned GPU window."""
import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from incremental import IncrementalBlockTables
from vllm.v1.worker.gpu.block_table import BlockTables


def test_locked_control_source():
    path = Path(inspect.getfile(BlockTables))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "e1597f285faaf62a84f2ea58ff88799d7c166273caa68c27134efdf5ce5e8004"
    )


class Case:
    def __init__(self, mode, concurrency, context):
        self.concurrency = concurrency
        self.rows = min(concurrency + 3, 16)
        self.active = list(range(concurrency))
        self.live = min(context // 256, 4095)
        args = dict(
            block_sizes=[256, 256, 256, 16],
            kernel_block_sizes=[64, 256, 256, 16],
            max_num_blocks_per_group=[4096, 4096, 1, 33],
            max_num_reqs=16, max_num_batched_tokens=128,
            device=torch.device("cuda"),
            slot_mapping_circular=[False, False, True, False],
        )
        self.control = BlockTables(**args)
        self.candidate = IncrementalBlockTables(**args, mode=mode)
        self.pointers = [x.data_ptr() for x in self.candidate.input_block_tables]
        self.ids = {}
        for slot in range(16):
            self.replace(slot, self.make_ids(slot, self.live, 0))
        self.publish()

    def make_ids(self, slot, live, generation):
        start = 100000 * (generation + 1) + 10000 * slot
        # Noncontiguous physical IDs, including null IDs among live pages.
        long = [0 if i % 17 == 0 else start + i * 3 for i in range(live)]
        return (long, long.copy(), [start + 3], [start + i for i in range(min(live, 33))])

    def replace(self, slot, ids):
        self.ids[slot] = ids
        for tables in (self.control, self.candidate):
            tables.append_block_ids(slot, ids, overwrite=True)

    def publish(self):
        for tables in (self.control, self.candidate):
            tables.apply_staged_writes()

    def check(self, active=None, rows=None, idx_dtype=torch.int32):
        active = self.active if active is None else active
        rows = self.rows if rows is None else rows
        idx = torch.tensor(active, dtype=idx_dtype, device="cuda")
        expected = self.control.gather_block_tables(idx, rows)
        actual = self.candidate.gather_block_tables(idx, rows)
        torch.cuda.synchronize()
        for ref, new in zip(expected, actual):
            assert torch.equal(ref, new)
        # Also compare untouched rows, and ensure capture-visible addresses stay.
        for ref, new in zip(self.control.input_block_tables, self.candidate.input_block_tables):
            assert torch.equal(ref, new)
        assert self.pointers == [x.data_ptr() for x in self.candidate.input_block_tables]


@pytest.mark.parametrize("mode", ["row", "chunk", "span"])
@pytest.mark.parametrize("concurrency", [1, 8, 16])
@pytest.mark.parametrize("context", [32768, 65536, 262144, 524288, 1048576])
def test_transitions(mode, concurrency, context):
    case = Case(mode, concurrency, context)
    case.check()  # Initial population.
    case.check()  # Growth inside a page and circular wrap: IDs stay unchanged.
    for slot in case.active:
        delta = ([900000 + slot], [800000 + slot], [], [])
        for tables in (case.control, case.candidate):
            tables.append_block_ids(slot, delta, overwrite=False)
    case.publish()
    case.check()  # Includes a crossing to exactly 1M, and 4x kernel-block expansion.
    case.check(list(reversed(case.active)))
    case.check([(slot + 1) % 16 for slot in case.active])
    case.check([], case.rows)  # Preempted rows become zero.
    case.check(case.active[: concurrency // 2], case.rows)
    for generation in (1, 2):  # Resume and same-size slot reuse.
        for slot in case.active:
            case.replace(slot, case.make_ids(slot, case.live, generation))
        case.publish()
        case.check()
    for slot in case.active:
        case.replace(slot, case.make_ids(slot, 1, 3))
    case.publish()
    case.check()  # Shorter reuse must clear every formerly live tail column.
    for tables in (case.control, case.candidate):
        tables.get_dummy_block_tables(case.rows)
    case.check()  # The source did not change, but dummy writes invalidated output.
    for slot in case.active:
        case.replace(slot, ([], [], [], []))
    case.publish()
    case.check()
    case.check([], 0)


@pytest.mark.parametrize("mode", ["row", "chunk", "span"])
def test_int64_mapping(mode):
    case = Case(mode, 8, 32768)
    case.check(idx_dtype=torch.int64)
    case.check(list(reversed(case.active)), idx_dtype=torch.int64)


@pytest.mark.parametrize("mode", ["row", "chunk", "span"])
def test_alternate_output_alias_and_layout_rebuild(mode):
    case = Case(mode, 8, 32768)
    case.check()
    idx = torch.tensor(list(reversed(case.active)), dtype=torch.int32, device="cuda")
    for tables in (case.control, case.candidate):
        tables.gather_block_tables(
            idx, case.rows, out=tuple(tables.input_block_tables),
            out_ptrs=tables.input_block_table_ptrs,
        )
    case.check()
    for tables in (case.control, case.candidate):
        tables.init_block_table_layout_tensors()
    case.check()


@pytest.mark.parametrize("mode", ["row", "chunk", "span"])
def test_graph_consumers_keep_stable_tables_across_shape_switches(mode):
    case = Case(mode, 8, 32768)
    # Capture two downstream integer consumers. This checks address/state
    # lifetime, not native-model arithmetic or a serving output digest.
    graphs = []
    for rows in (4, 8):
        case.candidate.get_dummy_block_tables(rows)
        output = torch.zeros((4, rows), dtype=torch.int64, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output.copy_(torch.stack([t[:rows].sum(1) for t in case.candidate.input_block_tables]))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output.copy_(torch.stack([t[:rows].sum(1) for t in case.candidate.input_block_tables]))
        graphs.append((rows, graph, output))
    # Both adapters now agree on dummy output before comparing untouched rows.
    case.control.get_dummy_block_tables(8)
    for generation in (1, 2, 3):
        for slot in case.active:
            case.replace(slot, case.make_ids(slot, case.live, generation))
        case.publish()
        for rows, graph, output in graphs:
            case.check(case.active[:rows], rows)
            graph.replay()
            expected = torch.stack([t[:rows].sum(1) for t in case.control.input_block_tables])
            assert torch.equal(output, expected)
