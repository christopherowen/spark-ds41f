"""CPU-only checks of the prototype's generation/publication methods and worker resource bounds.

CUDA-dependent construction is replaced with small fakes. These tests cannot
establish device correctness; that is the separate GPU gate.
"""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
RUN_SPEC = importlib.util.spec_from_file_location("block_metadata_run", HERE / "run.py")
run_mod = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run_mod)


class Base:
    def append_block_ids(self, req_index, new_block_ids, overwrite):
        self.events.append(("stage", req_index, overwrite))

    def apply_staged_writes(self):
        self.events.append("apply_ids_and_counts")


class Epochs:
    def __init__(self, events):
        self.np = [0, 0]
        self.events = events

    def copy_to_uva(self):
        self.events.append(("publish_epochs", self.np.copy()))


def prototype_class():
    module = ast.parse(Path(__file__).with_name("incremental.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ("append_block_ids", "apply_staged_writes")]
    namespace = {"BlockTables": Base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "incremental.py", "exec"), namespace)
    return namespace[cls.name]


class HostContract(unittest.TestCase):
    def make(self):
        obj = prototype_class()()
        obj.events = []
        obj.epochs = Epochs(obj.events)
        obj._epochs_dirty = False
        return obj

    def test_append_does_not_publish_a_replacement_generation(self):
        obj = self.make()
        obj.append_block_ids(0, ([1, 2],), False)
        obj.apply_staged_writes()
        self.assertEqual(obj.epochs.np, [0, 0])
        self.assertEqual(obj.events, [("stage", 0, False), "apply_ids_and_counts"])

    def test_same_count_and_empty_replacements_advance_generation(self):
        obj = self.make()
        for values in ([1], [2], []):
            obj.append_block_ids(1, (values,), True)
        self.assertEqual(obj.epochs.np, [0, 3])
        obj.apply_staged_writes()
        self.assertEqual(obj.events[-2:], ["apply_ids_and_counts", ("publish_epochs", [0, 3])])
        obj.apply_staged_writes()
        self.assertEqual(sum(isinstance(x, tuple) and x[0] == "publish_epochs" for x in obj.events), 1)

    def test_generation_exhaustion_fails_before_staging(self):
        obj = self.make()
        obj.epochs.np[0] = (1 << 63) - 1
        with self.assertRaises(OverflowError):
            obj.append_block_ids(0, ([3],), True)
        self.assertEqual(obj.events, [])


class WorkerBoundsContract(unittest.TestCase):
    def write_cgroup(
        self,
        root: Path,
        *,
        cpu_max: str = "200000 100000",
        memory_max: str = str(16 * 1024 ** 3),
        memory_swap_max: str = "0",
        pids_max: str = "512",
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "cpu.max").write_text(cpu_max + "\n")
        (root / "memory.max").write_text(memory_max + "\n")
        (root / "memory.swap.max").write_text(memory_swap_max + "\n")
        (root / "pids.max").write_text(pids_max + "\n")
        return root

    def test_rendered_launch_applies_container_and_thread_bounds_across_both_boots(self):
        plan = run_mod.render_launch_plan(HERE, cache_dir="/tmp/spark-cache")
        self.assertEqual([item["process"] for item in plan], [1, 2])
        for boot, item in zip((1, 2), plan):
            cmd = item["command"]
            for required in (
                "--cpus=2",
                "--memory=16g",
                "--memory-swap=16g",
                "--pids-limit=512",
                "--gpus",
                "all",
                "--ipc=host",
                "--ulimit=core=0",
                run_mod.EXPECTED_IMAGE,
                "/b/run.py",
                "--worker",
            ):
                self.assertIn(required, cmd)
            self.assertEqual(cmd[-2:], ["--boot", str(boot)])
            for key, limit in run_mod.THREAD_LIMITS.items():
                self.assertIn(f"{key}={limit}", cmd)
            mounts = [cmd[i + 1] for i, token in enumerate(cmd[:-1]) if token == "-v"]
            bundle_mounts = [m for m in mounts if ":/b/" in m]
            self.assertEqual(len(bundle_mounts), 6)
            self.assertTrue(all(m.endswith(":ro") for m in bundle_mounts))

    def test_worker_cgroup_and_thread_bounds_fail_closed_when_unbounded_or_excessive(self):
        candidate = run_mod.load_candidate()
        base_env = run_mod.worker_environment(candidate, {})
        with tempfile.TemporaryDirectory() as tmp:
            cgroup = self.write_cgroup(Path(tmp))
            verified = run_mod.verify_worker_bounds(cgroup, base_env, candidate)
            self.assertEqual(verified["cpus"], 2.0)
            self.assertEqual(verified["memory_max_bytes"], 16 * 1024 ** 3)
            self.assertEqual(verified["memory_swap_max_bytes"], 0)
            self.assertEqual(verified["pids_max"], 512)
            self.assertEqual(verified["thread_limits"], run_mod.THREAD_LIMITS)

            invalid_cgroups = [
                {"cpu_max": "max 100000"},
                {"cpu_max": "300000 100000"},
                {"memory_max": "max"},
                {"memory_max": str(32 * 1024 ** 3)},
                {"memory_swap_max": "max"},
                {"memory_swap_max": "1048576"},
                {"pids_max": "max"},
                {"pids_max": "1024"},
            ]
            for override in invalid_cgroups:
                with self.subTest(cgroup_override=override):
                    self.write_cgroup(cgroup, **override)
                    with self.assertRaises(RuntimeError):
                        run_mod.verify_worker_bounds(cgroup, base_env, candidate)
            self.write_cgroup(cgroup)

            for key, limit in run_mod.THREAD_LIMITS.items():
                for bad_value in ("", "0", str(limit + 1), "unlimited"):
                    with self.subTest(env_key=key, bad_value=bad_value):
                        bad_env = dict(base_env)
                        bad_env[key] = bad_value
                        with self.assertRaises(RuntimeError):
                            run_mod.verify_worker_bounds(cgroup, bad_env, candidate)

    def test_worker_preserves_bounds_across_both_fresh_boots_and_records_receipts(self):
        candidate = run_mod.load_candidate()
        base_env = run_mod.worker_environment(candidate, {})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cgroup = self.write_cgroup(root / "cgroup")
            cgroup.mkdir(parents=True, exist_ok=True)
            self.write_cgroup(cgroup)
            out = root / "out-split"
            calls = []

            def fake_runner(argv, *, env, timeout):
                calls.append((argv, dict(env), timeout))
                return SimpleNamespace(returncode=0)

            run_mod.run_worker(out, boots=(1,), cgroup_root=cgroup, environ=base_env, runner=fake_runner)
            run_mod.run_worker(out, boots=(2,), cgroup_root=cgroup, environ=base_env, runner=fake_runner)
            self.assertEqual(len(calls), 6)
            steps = json.loads((out / "steps.json").read_text())
            self.assertEqual([s["process"] for s in steps], [1, 1, 1, 2, 2, 2])
            self.assertEqual(
                [s["step"] for s in steps],
                [
                    "host-publication",
                    "GPU-transition-gates",
                    "component-benchmark",
                    "host-publication",
                    "GPU-transition-gates",
                    "component-benchmark",
                ],
            )
            for (_, call_env, _), step in zip(calls, steps):
                for key, limit in run_mod.THREAD_LIMITS.items():
                    self.assertEqual(int(call_env[key]), limit)
                self.assertEqual(step["bounds"]["cpus"], 2.0)
                self.assertEqual(step["bounds"]["memory_max_bytes"], 16 * 1024 ** 3)
                self.assertEqual(step["bounds"]["memory_swap_max_bytes"], 0)
                self.assertEqual(step["bounds"]["pids_max"], 512)

            unbounded = root / "cgroup-unbounded"
            unbounded.mkdir(parents=True, exist_ok=True)
            self.write_cgroup(unbounded, memory_swap_max="max")
            blocked_out = root / "must-not-exist"
            blocked_calls = []
            with self.assertRaises(RuntimeError):
                run_mod.run_worker(
                    blocked_out,
                    boots=(1, 2),
                    cgroup_root=unbounded,
                    environ=base_env,
                    runner=lambda *a, **k: blocked_calls.append((a, k)),
                )
            self.assertEqual(blocked_calls, [])
            self.assertFalse(blocked_out.exists())

    def test_host_controller_enforces_exact_window_ownership_and_image_identity(self):
        valid_hold = {
            "holder": run_mod.DEFAULT_HOLDER,
            "window": "dgx1-owned",
            "production_config": run_mod.DEFAULT_RESTORE,
        }
        for bad_hold in (
            None,
            {},
            {**valid_hold, "holder": "spark-lab: other task"},
            {**valid_hold, "window": "dgx1-stale"},
            {**valid_hold, "production_config": "other.json"},
        ):
            with self.subTest(bad_hold=bad_hold):
                with self.assertRaises(RuntimeError):
                    run_mod.require_exact_owner(bad_hold, window="dgx1-owned")

        holds = iter([valid_hold, valid_hold, {**valid_hold, "holder": "spark-lab: foreign"}])
        dispatched = []

        def fake_host_runner(argv, **kwargs):
            dispatched.append(argv)
            if argv[:3] == ["docker", "image", "inspect"]:
                labels = json.dumps({"local.spark-ds41f.vllm.tree": run_mod.EXPECTED_TREE})
                return SimpleNamespace(returncode=0, stdout=f"{run_mod.EXPECTED_IMAGE} {labels}\n", stderr="")
            return SimpleNamespace(returncode=0)

        with self.assertRaises(RuntimeError):
            run_mod.run_host(
                hold_loader=lambda: next(holds),
                window="dgx1-owned",
                runner=fake_host_runner,
            )
        # Inspect ran, boot 1 ran, boot 2 was blocked before docker run when ownership changed.
        self.assertEqual(len(dispatched), 2)
        self.assertEqual(dispatched[1][-2:], ["--boot", "1"])


if __name__ == "__main__":
    unittest.main()
