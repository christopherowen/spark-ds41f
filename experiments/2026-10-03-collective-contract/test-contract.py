"""Exercise actual adapter/dispatch methods without CUDA imports or a live cluster.

Only dependency boundaries are mocked. AST extraction keeps tests usable on a
CPU host; it does not substitute a reimplementation of the selection logic.
"""

# ruff: noqa: S102 -- execute only the explicitly supplied, trusted source tree
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

if len(sys.argv) > 1:
    SOURCE = Path(sys.argv.pop(1)).resolve()
else:
    import argparse
    import importlib.machinery
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    loader = importlib.machinery.SourceFileLoader(
        "contract_build", str(root / "bin/spark3")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    build = importlib.util.module_from_spec(spec)
    loader.exec_module(build)
    _, _, lock = build.configuration(
        argparse.Namespace(
            cluster_config="experiments/2026-10-03-collective-contract/candidate.json"
        )
    )
    inputs = build.build_inputs(lock)
    source_root = build.build_directory(inputs) / "src"
    for name in ("vllm", "b12x"):
        problems = build.project_problems(source_root / name, inputs[name])
        if problems:
            raise SystemExit("\n".join(problems))
    SOURCE = source_root / "vllm"
COMM = SOURCE / "vllm/distributed/device_communicators"


def extract(path, name, members, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    nodes = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in members
    ]
    cls.bases = []
    cls.decorator_list = []
    cls.body = nodes
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Tensor:
    dtype = "bfloat16"
    is_cuda = True
    device = "cuda:0"
    is_sparse = False

    def __init__(self, shape, dtype="bfloat16", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.contiguous_flag = contiguous

    def numel(self):
        import math

        return math.prod(self.shape)

    def element_size(self):
        return 4 if self.dtype == "float32" else 2

    def size(self):
        return self.shape

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous_flag

    def is_complex(self):
        return False

    def contiguous(self):
        return self

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor(shape)

    def movedim(self, a, b):
        shape = list(self.shape)
        shape.insert(b if b >= 0 else len(shape) - 1, shape.pop(a))
        return Tensor(shape)


class Contract(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.namespace = {
            "logger": self.logger,
            "dist": types.SimpleNamespace(
                get_rank=lambda **kw: 0, get_world_size=lambda **kw: 4
            ),
            "in_the_same_node_as": lambda *a, **kw: [True, False, False, False],
            "register_b12x_unit_provider": Mock(),
        }
        self.Adapter = extract(
            COMM / "b12x_roce_all_reduce.py",
            "B12xRoceAllReduce",
            {
                "__init__",
                "_exchange_vote",
                "all_reduce_max_bytes",
                "all_reduce_capacity_bytes",
                "all_gather_max_bytes",
                "should_custom_ar",
                "custom_all_reduce",
                "should_all_gather",
                "all_gather",
            },
            self.namespace,
        )
        b12x_path = SOURCE.parent / "b12x/b12x/comm/roce/roce_oneshot.py"
        if not b12x_path.exists():
            b12x_path = (
                SOURCE.parent
                / "rocenante-bidirectional-src/b12x/comm/roce/roce_oneshot.py"
            )
        Runtime = extract(
            b12x_path,
            "RoceOneshotAllReduce",
            {
                "should_allreduce",
                "_accepts_allreduce",
                "should_all_gather",
                "_normalize_dim",
            },
            {
                "SUPPORTED_DTYPES": ("float16", "bfloat16", "float32"),
                "PACK_BYTES": 16,
                "torch": types.SimpleNamespace(bool="bool"),
            },
        )
        self.runtime = Runtime()
        self.runtime.__dict__.update(
            _closed=False,
            _proxy=object(),
            device="cuda:0",
            max_size=2097152,
            dispatch_max_bytes=1048576,
            max_gather_bytes=2097152,
            hca_names=["a", "b", "c", "d"],
            all_reduce=Mock(return_value="roce"),
            all_gather=Mock(return_value="roce-gather"),
        )
        self.adapter = self.Adapter.__new__(self.Adapter)
        self.adapter.__dict__.update(
            disabled=False,
            _runtime=self.runtime,
            rank=0,
            _announced=False,
            _announced_gather=False,
            _prepared_plan=lambda: object(),
        )
        self.torch = types.SimpleNamespace(
            Tensor=Tensor, empty=lambda shape, **kw: Tensor(shape)
        )
        namespace = {
            "torch": self.torch,
            "logger": self.logger,
            "should_nccl_symm_mem_ag_rs": Mock(
                side_effect=AssertionError("unplanned symmetric path")
            ),
            "envs": types.SimpleNamespace(VLLM_BATCH_INVARIANT=False),
            "current_platform": types.SimpleNamespace(is_rocm=lambda: False),
        }
        Cuda = extract(
            COMM / "cuda_communicator.py",
            "CudaCommunicator",
            {
                "_require_roce_policy",
                "all_reduce",
                "all_gather",
                "reduce_scatter",
                "reduce_scatterv",
                "all_gatherv",
                "_log_all_reduce_backend_selection",
            },
            namespace,
        )
        self.comm = Cuda()
        self.nccl = types.SimpleNamespace(
            disabled=False,
            all_reduce=Mock(return_value="nccl"),
            all_gather=Mock(),
            reduce_scatter=Mock(),
            all_gatherv=Mock(),
            reduce_scatterv=Mock(),
            reduce=Mock(),
            scatter=Mock(),
        )
        self.comm.__dict__.update(
            _roce_policy=True,
            b12x_ar_comm=self.adapter,
            pynccl_comm=self.nccl,
            world_size=4,
            rank_in_group=0,
            unique_name="tp:0",
        )
        self.comm._can_use_aiter_ag_rs = Mock(
            side_effect=AssertionError("unplanned AITER path")
        )

    def test_dispatch_boundary_and_capacity_are_distinct(self):
        for dtype in ("float16", "bfloat16", "float32"):
            width = 4 if dtype == "float32" else 2
            for size, backend in [
                (16, "roce"),
                (1048576, "roce"),
                (1048592, "nccl"),
                (2097152, "nccl"),
            ]:
                with self.subTest(dtype=dtype, size=size):
                    self.assertEqual(
                        self.comm.all_reduce(Tensor([size // width], dtype)), backend
                    )
        self.assertEqual(self.adapter.all_reduce_max_bytes, 1048576)
        self.assertEqual(self.adapter.all_reduce_capacity_bytes, 2097152)
        self.assertIn(
            "dispatch limit=1048576 bytes, registered capacity=2097152 bytes",
            self.logger.info.call_args.args[0] % self.logger.info.call_args.args[1:],
        )

    def test_ineligible_inputs_select_nccl(self):
        for inp in [
            Tensor([1]),
            Tensor([0]),
            Tensor([8], "int64"),
            Tensor([8], contiguous=False),
        ]:
            self.assertEqual(self.comm.all_reduce(inp), "nccl")

    def test_no_error_fallback(self):
        self.runtime.all_reduce.side_effect = RuntimeError("transport fault")
        with self.assertRaisesRegex(RuntimeError, "transport fault"):
            self.comm.all_reduce(Tensor([8]))
        self.nccl.all_reduce.assert_not_called()
        self.runtime.all_reduce.side_effect = None
        self.runtime.all_reduce.return_value = None
        with self.assertRaisesRegex(RuntimeError, "B12X all-reduce returned None"):
            self.comm.all_reduce(Tensor([8]))
        self.nccl.all_reduce.return_value = None
        with self.assertRaisesRegex(RuntimeError, "returned None"):
            self.comm.all_reduce(Tensor([1048576]))

    def test_closed_runtime_fails(self):
        self.runtime._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.comm.all_reduce(Tensor([8]))
        self.nccl.all_reduce.assert_not_called()

    def test_disabled_backends_fail_for_every_collective(self):
        calls = [
            ("all_reduce", (Tensor([8]),)),
            ("all_gather", (Tensor([8]),)),
            ("reduce_scatter", (Tensor([8]),)),
            ("reduce_scatterv", (Tensor([8]),)),
            ("all_gatherv", (Tensor([8]),)),
        ]
        for missing in ("roce", "nccl"):
            self.adapter.disabled = missing == "roce"
            self.nccl.disabled = missing == "nccl"
            for method, args in calls:
                with (
                    self.subTest(missing=missing, method=method),
                    self.assertRaisesRegex(RuntimeError, "lost"),
                ):
                    getattr(self.comm, method)(*args)

    def test_gather_cutoff_is_input_shard(self):
        self.assertEqual(self.comm.all_gather(Tensor([1048576])), "roce-gather")
        out = self.comm.all_gather(Tensor([1048584]))
        self.assertEqual(out.shape, (4194336,))
        self.nccl.all_gather.assert_called_once()
        self.assertEqual(self.adapter.all_gather_max_bytes, 2097152)

    def test_scatter_and_variable_collectives_use_nccl(self):
        self.assertEqual(self.comm.reduce_scatter(Tensor([16])).shape, (4,))
        self.comm.reduce_scatterv(Tensor([16]))
        self.comm.all_gatherv(Tensor([4]))
        self.comm.reduce_scatterv(Tensor([16]), sizes=[3, 4, 4, 5])
        self.comm.all_gatherv(Tensor([3]), sizes=[3, 4, 4, 5])
        self.assertEqual(self.nccl.reduce_scatter.call_count, 2)
        self.nccl.reduce_scatterv.assert_called_once()
        self.nccl.all_gather.assert_called_once()
        self.nccl.all_gatherv.assert_called_once()

    def test_backend_log_only_names_actual_policy(self):
        self.comm._log_all_reduce_backend_selection()
        line = (
            self.logger.info_once.call_args.args[0]
            % self.logger.info_once.call_args.args[1:]
        )
        self.assertIn("B12X_ROCENANTE", line)
        self.assertIn("PYNCCL", line)
        self.assertNotIn("FLASHINFER", line)

    def construct(self, *, vote=None, nccl=True, fault=None):
        self.Adapter._local_capability = lambda obj: (None, (2097152, 2097152))
        self.Adapter._exchange_vote = lambda obj, reason, limits: reason or vote
        factory = Mock(return_value=self.runtime, side_effect=fault)
        roce = types.SimpleNamespace(
            AllReduce=types.SimpleNamespace(from_exchange_group=factory)
        )
        with patch.dict(
            sys.modules,
            {
                "b12x": types.ModuleType("b12x"),
                "b12x.comm": types.SimpleNamespace(roce=roce),
            },
        ):
            return self.Adapter("group", "devicegroup", "cuda:0", nccl_available=nccl)

    def test_startup_fails_for_vote_nccl_and_constructor_errors(self):
        for kwargs, message in [
            ({"vote": "rank 2: unsupported"}, "rank 2"),
            ({"nccl": False}, "PyNCCL"),
            ({"fault": RuntimeError("fault")}, "initialization failed"),
        ]:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.construct(**kwargs)

    def test_successful_startup_reports_runtime_policy(self):
        obj = self.construct()
        self.assertFalse(obj.disabled)
        line = self.logger.info.call_args.args[0] % self.logger.info.call_args.args[1:]
        for part in (
            "dispatch <=1048576",
            "capacity=2097152",
            "input shard <=2097152",
            "failures are fatal",
        ):
            self.assertIn(part, line)

    def test_peer_vote_rejects_missing_or_different_capabilities(self):
        obj = self.Adapter.__new__(self.Adapter)
        obj.world_size = 4
        obj.group = "cpu-group"
        limits = (2097152, 2097152)
        votes = [(None, limits)] * 4
        self.namespace["dist"].all_gather_object = lambda out, local, **kw: (
            out.__setitem__(slice(None), votes)
        )
        self.assertIsNone(obj._exchange_vote(None, limits))
        votes[2] = ("missing package", None)
        self.assertIn("rank 2: missing package", obj._exchange_vote(None, limits))
        votes[2] = (None, (1048576, 2097152))
        self.assertIn("size limits differ", obj._exchange_vote(None, limits))

    def test_disabled_adapter_reports_zero_limits(self):
        self.adapter.disabled = True
        self.assertEqual(self.adapter.all_reduce_max_bytes, 0)
        self.assertEqual(self.adapter.all_reduce_capacity_bytes, 0)
        self.assertEqual(self.adapter.all_gather_max_bytes, 0)

    def test_single_node_keeps_existing_backend(self):
        self.namespace["in_the_same_node_as"] = lambda *a, **kw: [True] * 4
        self.assertTrue(self.construct(nccl=False).disabled)

    def test_capacity_keeps_sp_boundary(self):
        namespace = {
            "model_parallel_is_initialized": lambda: True,
            "get_tp_group": lambda: types.SimpleNamespace(
                device_communicator=self.comm
            ),
            "logger": self.logger,
        }
        tree = ast.parse(
            (SOURCE / "vllm/models/deepseek_v4_1/sp_prefill.py").read_text()
        )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_prefill_sp_capacity_bytes",
                "rows_above_bytes",
                "resolve_min_rows",
            ):
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), "<actual-sp>", "exec"
                    ),
                    namespace,
                )
        S = types.SimpleNamespace
        config = S(
            parallel_config=S(tensor_parallel_size=4, pipeline_parallel_size=1),
            use_request_boundary_checkpoints=False,
            speculative_config=S(num_speculative_tokens=5),
            compilation_config=S(max_cudagraph_capture_size=48),
            scheduler_config=S(max_num_seqs=8),
            model_config=S(get_hidden_size=lambda: 5120, dtype=S(itemsize=2)),
        )
        for dispatch in [1048576, 2097152]:
            self.runtime.dispatch_max_bytes = dispatch
            self.assertEqual(namespace["resolve_min_rows"](config), 205)


if __name__ == "__main__":
    unittest.main(verbosity=2)
