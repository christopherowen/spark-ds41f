"""Verdict statistics: averaging both boots on each side, the boot term, recorded floors by
testbed and suite, the startup-memory summary's start, and first boots (CPU only)."""

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from servebench import __main__ as cli  # noqa: E402
from servebench import suite  # noqa: E402
from spark_ds41f import candidate, lab  # noqa: E402

_spec = importlib.util.spec_from_file_location("servebench_suite_table", ROOT / "scripts/servebench_suite_table.py")
table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table)


def paired(change: float, half: float = 0.05) -> dict:
    return {"metric": "decode tok/s, paired", "change_pct": change, "ci95_pct": half,
            "low_pct": change - half, "high_pct": change + half, "higher_is_better": True}


def outcome(grid: float, tps: float, pair: float | None = None, grid_ci: float = 0.15) -> dict:
    rows = [{"metric": "output tok/s", "change_pct": tps, "ci95_pct": 0.9}]
    if pair is not None:
        rows.append(paired(pair))
    return {"steps": {"grid": ("servebench-engine/1", {"rows": [
                {"metric": "decode geometric mean (20)", "change_pct": grid, "ci95_pct": grid_ci}]}),
            "c16": ("servebench/1", {"rows": rows})}}


class BootTermTest(unittest.TestCase):
    def test_one_bound_for_both_sides(self):
        self.assertAlmostEqual(suite.boot_term(0.4, None, 1, 1), 0.4)
        self.assertAlmostEqual(suite.boot_term(0.4, None, 2, 1), 0.4 * math.sqrt(0.75))
        self.assertAlmostEqual(suite.boot_term(0.4, 0.4, 2, 2), 0.4 * math.sqrt(0.5))
        self.assertAlmostEqual(suite.boot_term(0.4, None, 2, 2), 0.4 * math.sqrt(0.5))

    def test_each_side_its_own_bound(self):
        self.assertAlmostEqual(suite.boot_term(0.4, 0.2, 2, 2), math.sqrt(0.4 ** 2 / 4 + 0.2 ** 2 / 4))
        self.assertAlmostEqual(suite.boot_term(0.4, 0.2, 1, 2), math.sqrt(0.4 ** 2 / 2 + 0.2 ** 2 / 4))


class AveragingTest(unittest.TestCase):
    floors = {"c16": 0.43, "grid": 0.43}

    def rows(self, **extra):
        return {(r["step"], r["metric"]): r for r in suite.with_floor(
            outcome(-1.0, 1.0, 0.40), outcome(0.10, -0.10, 0.10), floors=self.floors,
            reference_to_repeat=outcome(-0.9, 1.2, 0.30), **extra)}

    def test_without_a_reference_repeat_nothing_changes(self):
        rows = self.rows()
        pair = rows[("c16", "decode tok/s, paired")]
        self.assertEqual((pair["boots"], pair["reference_boots"]), (2, 1))
        self.assertAlmostEqual(pair["change_pct"], 0.35)
        self.assertAlmostEqual(pair["floor_pct"], 0.43 * math.sqrt(0.75))
        self.assertIsNone(pair["reference_repeat_pct"])
        # A window row keeps the first pairing and interval + |repeat|.
        window = rows[("c16", "output tok/s")]
        self.assertEqual((window["boots"], window["reference_boots"], window["change_pct"]), (1, 1, 1.0))
        self.assertAlmostEqual(window["margin_pct"], 0.9 + 0.10)
        # The grid's geometric mean takes its recorded floor over both candidate boots.
        grid = rows[("grid", "decode geometric mean (20)")]
        self.assertEqual(grid["boots"], 2)
        self.assertAlmostEqual(grid["change_pct"], -0.95)
        self.assertAlmostEqual(grid["floor_pct"], 0.43 * math.sqrt(0.75))
        self.assertAlmostEqual(grid["margin_pct"], math.hypot(0.15, 0.43 * math.sqrt(0.75)))
        self.assertEqual(grid["verdict"], "better")  # step times fell

    def test_a_grid_without_a_floor_keeps_interval_plus_repeat(self):
        rows = {(r["step"], r["metric"]): r for r in suite.with_floor(outcome(-0.3, 1.0), outcome(0.2, 0.1))}
        grid = rows[("grid", "decode geometric mean (20)")]
        self.assertIsNone(grid["floor_pct"])
        self.assertAlmostEqual(grid["margin_pct"], 0.15 + 0.2)
        self.assertEqual(grid["verdict"], "same")

    def test_both_boots_of_each(self):
        rows = self.rows(reference_repeat=outcome(0.05, 0.2, -0.1),
                         reference_again_to_candidate=outcome(-1.1, 0.8, 0.50),
                         reference_again_to_repeat=outcome(-1.0, 1.0, 0.40))
        pair = rows[("c16", "decode tok/s, paired")]
        self.assertEqual((pair["boots"], pair["reference_boots"]), (2, 2))
        self.assertAlmostEqual(pair["change_pct"], (0.40 + 0.30 + 0.50 + 0.40) / 4)
        self.assertAlmostEqual(pair["reference_repeat_pct"], -0.1)
        self.assertAlmostEqual(pair["floor_pct"], 0.43 * math.sqrt(0.5))
        self.assertAlmostEqual(pair["margin_pct"], math.hypot(0.05, 0.43 * math.sqrt(0.5)))
        window = rows[("c16", "output tok/s")]
        self.assertAlmostEqual(window["change_pct"], (1.0 + 0.8 + 1.2 + 1.0) / 4)
        # Window rows: interval + the repeats' boot term over two boots of each.
        self.assertAlmostEqual(window["boot_pct"], math.sqrt(0.10 ** 2 / 4 + 0.2 ** 2 / 4))
        self.assertAlmostEqual(window["margin_pct"], 0.9 + window["boot_pct"])
        grid = rows[("grid", "decode geometric mean (20)")]
        self.assertAlmostEqual(grid["change_pct"], (-1.0 - 0.9 - 1.1 - 1.0) / 4)
        self.assertAlmostEqual(grid["floor_pct"], 0.43 * math.sqrt(0.5))

    def test_a_noisy_repeat_on_either_side_widens_its_bound(self):
        rows = self.rows(reference_repeat=outcome(0.9, 0.2, -0.8),
                         reference_again_to_candidate=outcome(-1.1, 0.8, 0.50),
                         reference_again_to_repeat=outcome(-1.0, 1.0, 0.40))
        self.assertAlmostEqual(rows[("c16", "decode tok/s, paired")]["floor_pct"], math.sqrt(0.43 ** 2 / 4 + 0.8 ** 2 / 4))
        self.assertAlmostEqual(rows[("grid", "decode geometric mean (20)")]["floor_pct"],
                               math.sqrt(0.43 ** 2 / 4 + 0.9 ** 2 / 4))

    def test_a_missing_pairing_counts_only_the_boots_used(self):
        rows = self.rows(reference_repeat=outcome(0.05, 0.2, -0.1), reference_again_to_candidate=outcome(-1.1, 0.8, 0.5))
        pair = rows[("c16", "decode tok/s, paired")]
        self.assertEqual((pair["boots"], pair["reference_boots"]), (2, 2))
        self.assertAlmostEqual(pair["change_pct"], (0.40 + 0.50 + 0.30) / 3)

    def test_the_table_says_which_boots_were_averaged(self):
        rows = suite.with_floor(outcome(-1.0, 1.0, 0.4), outcome(0.1, -0.1, 0.1), floors=self.floors,
                                reference_to_repeat=outcome(-0.9, 1.2, 0.3), reference_repeat=outcome(0.05, 0.2, -0.1),
                                reference_again_to_candidate=outcome(-1.1, 0.8, 0.5),
                                reference_again_to_repeat=outcome(-1.0, 1.0, 0.4))
        text = suite.format_floor(rows, "ref", "cand", "cand-again", reference_repeat="ref-again",
                                  floors_note="floors (tp4-ring, screen): grid 0.37% (6 pairs)")
        head = text.splitlines()[0]
        self.assertTrue(head.startswith("verdicts: ref -> cand, against the run's own repeat (cand -> cand-again)"))
        self.assertIn("and the reference's (ref-again); each change is the mean over both boots of each "
                      "(cand, cand-again; ref, ref-again)", head)
        self.assertIn("(both boots of each, reference repeat -0.10%; window tok/s", text)
        self.assertIn("floors (tp4-ring, screen)", text)
        self.assertTrue(text.splitlines()[-1].startswith("paired rows: margin"))
        self.assertIn("in quadrature", text)
        # Without a reference repeat the header is the old one.
        plain = suite.format_floor(suite.with_floor(outcome(-1.0, 1.0, 0.4), outcome(0.1, -0.1, 0.1), floors=self.floors),
                                   "ref", "cand", "cand-again")
        self.assertEqual(plain.splitlines()[0], "verdicts: ref -> cand, against the run's own repeat (cand -> cand-again)")


def index(name: str, transport: str | None, nodes: int, steps: list[dict], digest: str = "d" * 64) -> dict:
    identity = {"transport": transport, "topology": {"nodes": [{"name": f"n{i}"} for i in range(nodes)]}} if transport else {}
    return {"suite": {"name": name, "steps": steps}, "suite_sha256": digest, "metadata": {"identity": identity}}


SCREEN = [{"id": "model", "command": "model"}, {"id": "grid", "command": "engine"}, {"id": "c16", "command": "run"},
          {"id": "prefill", "command": "prefill"}]


class FloorLookupTest(unittest.TestCase):
    def test_testbed_from_the_recorded_identity(self):
        self.assertEqual(table.testbed_of(index("screen", "oneshot-ring4", 4, SCREEN)), "tp4-ring")
        self.assertEqual(table.testbed_of(index("screen", "oneshot-switched", 4, SCREEN)), "tp4-switched")
        self.assertEqual(table.testbed_of(index("screen-tp3", "oneshot-switched", 3, SCREEN)), "tp3-switched")
        self.assertEqual(table.testbed_of(index("screen-tp3", "oneshot-direct", 3, SCREEN)), "tp3-triangle")
        self.assertIsNone(table.testbed_of(index("screen", None, 0, SCREEN)))

    def test_a_testbed_and_suite_take_their_own_floors(self):
        floors, note = table.floors_for(index("screen", "oneshot-ring4", 4, SCREEN))
        self.assertEqual(floors, {"grid": 0.37, "c16": 0.39})
        self.assertIn("floors (tp4-ring, screen): grid 0.37% (6 pairs); c16 0.39% (6 pairs)", note)
        floors, _ = table.floors_for(index("screen", "oneshot-switched", 4, SCREEN))
        self.assertEqual(floors, {"grid": 0.33, "c16": 0.34})

    def test_too_few_pairs_keep_the_default_and_say_so(self):
        steps = [{"id": "grid", "command": "engine"}, {"id": "c8", "command": "run"}]
        floors, note = table.floors_for(index("screen-tp3", "oneshot-switched", 3, steps))
        self.assertEqual(floors, {"grid": table.GRID_FLOOR_PCT, "c8": table.AA_FLOOR_PCT["c8"]})
        self.assertIn("grid default 0.43% kept (5 pairs", note)
        self.assertIn("c8 default 0.39% kept (3 pairs, 0.52%)", note)
        single = [{"id": "c1", "command": "run"}]
        floors, note = table.floors_for(index("single-stream", "oneshot-ring4", 4, single))
        self.assertEqual(floors, {"c1": 0.35})
        self.assertIn("c1 default 0.35% kept (4 pairs", note)

    def test_more_pairs_can_raise_a_floor(self):
        sampled = [{"id": s, "command": "run"} for s in ("c16", "c8", "c1")]
        floors, _ = table.floors_for(index("sampled", "oneshot-switched", 3, sampled))
        self.assertEqual(floors, {"c16": 0.96, "c8": 0.39, "c1": 0.35})

    def test_an_unknown_testbed_takes_the_defaults(self):
        floors, note = table.floors_for(index("release", None, 0, [{"id": "grid", "command": "engine"},
                                                                   {"id": "c16", "command": "run"},
                                                                   {"id": "poisson", "command": "run"}]))
        self.assertEqual(floors, {"grid": table.GRID_FLOOR_PCT, "c16": 0.43})
        self.assertIn("testbed not recorded", note)
        self.assertIn("grid default 0.43%; c16 default 0.43%; the recorded defaults", note)
        _, none = table.floors_for(index("stall-tp3", "oneshot-switched", 3, [{"id": "stall-3", "command": "stall"}]))
        self.assertTrue(none.endswith("no step of this suite takes a recorded floor"))

    def test_every_recorded_floor_follows_the_rule(self):
        for key, entry in table.FLOORS.items():
            for step, value in entry["floors"].items():
                default = table.GRID_FLOOR_PCT if step == "grid" else table.AA_FLOOR_PCT[step]
                pairs = entry["pairs"][step]
                with self.subTest(key=key, step=step):
                    self.assertGreaterEqual(pairs, 4)
                    if value < default:
                        self.assertGreaterEqual(pairs, 6)


class ReferenceRepeatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_dir(self, path: Path, name: str = "screen", digest: str = "d" * 64) -> None:
        path.mkdir(parents=True)
        (path / "suite.json").write_text(json.dumps(index(name, "oneshot-ring4", 4, SCREEN, digest)))

    def test_names_a_repeat_takes(self):
        self.assertEqual(table.repeat_names("refs/c54-tp4-ring-screen", "screen"),
                         ["refs/c54-tp4-ring-screen-again", "refs/c54-tp4-ring-again-screen"])
        self.assertIn("refs/c54-tp3-switched-screen-again-single-stream",
                      table.repeat_names("refs/c54-tp3-switched-screen-single-stream", "single-stream"))
        self.assertIn("candidate-54-again-stall", table.repeat_names("candidate-54-stall", "stall-tp3"))

    def test_an_arm_named_after_the_reference(self):
        for label in ("cand", "ref", "ref-again", "cand-again"):
            self.run_dir(self.root / "run" / label)
        labels = ["cand", "ref", "ref-again", "cand-again"]
        self.assertEqual(table.reference_repeat("run", labels, self.root), ("ref-again", None))
        # No arm of that name: one reference boot, as before.
        self.assertEqual(table.reference_repeat("run", ["cand", "ref", "other", "cand-again"], self.root), (None, None))
        # Three labels: the last is the candidate's repeat, never the reference's.
        self.assertEqual(table.reference_repeat("run", ["cand", "ref", "ref-again"], self.root), (None, None))

    def test_a_record_beside_a_recorded_reference(self):
        refs = self.root / "repository"
        original, table.ROOT = table.ROOT, refs
        self.addCleanup(setattr, table, "ROOT", original)
        base = Path("experiments/x/references")
        self.run_dir(refs / base / "c54-tp3-switched-screen-single-stream", "single-stream")
        for label in ("cand", "cand-again"):
            self.run_dir(self.root / "run" / label, "single-stream")
        labels = ["cand", str(base / "c54-tp3-switched-screen-single-stream"), "cand-again"]
        self.assertEqual(table.reference_repeat("run", labels, self.root), (None, None))
        # A record of another suite definition is not averaged, and the table says why.
        self.run_dir(refs / base / "c54-tp3-switched-screen-single-stream-again", "single-stream", "e" * 64)
        found, why = table.reference_repeat("run", labels, self.root)
        self.assertIsNone(found)
        self.assertIn("not the reference's; not averaged", why)
        # The companion's record by the lab's naming is found, past the mismatched one.
        self.run_dir(refs / base / "c54-tp3-switched-screen-again-single-stream", "single-stream")
        self.assertEqual(table.reference_repeat("run", labels, self.root),
                         (str(base / "c54-tp3-switched-screen-again-single-stream"), None))

    def test_an_explicit_repeat(self):
        for label in ("cand", "ref", "second", "cand-again"):
            self.run_dir(self.root / "run" / label)
        labels = ["cand", "ref", "second", "cand-again"]
        self.assertEqual(table.reference_repeat("run", labels, self.root, "second"), ("second", None))
        found, why = table.reference_repeat("run", labels, self.root, "absent")
        self.assertIsNone(found)
        self.assertIn("no suite.json", why)


class TableEndToEndTest(unittest.TestCase):
    """Four arms of a tiny suite on a fake server: the reference's repeat is averaged."""

    def test_reference_repeat_arm_is_averaged(self):
        from tests.test_servebench import FakeServer, SuiteTest

        server = FakeServer()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                definition = os.path.join(tmp, "tiny.json")
                Path(definition).write_text(json.dumps(SuiteTest.TINY))
                metadata = os.path.join(tmp, "metadata.json")
                Path(metadata).write_text(json.dumps({"identity": {"transport": "oneshot-ring4",
                                                                   "topology": {"nodes": [{}, {}, {}, {}]}}}))
                runs = Path(tmp) / "suites" / "run"
                labels = ["cand", "ref", "ref-again", "cand-again"]
                for label in labels:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(cli.main(["suite", server.url, "--suite", definition, "--out", str(runs / label),
                                                   "--label", label, "--metadata", metadata]), 0)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    text = table.verdicts("run", labels, Path(tmp) / "suites")
                    plain = table.verdicts("run", labels, Path(tmp) / "suites", average_reference=False)
        finally:
            server.close()
        self.assertIn("and the reference's (ref-again); each change is the mean over both boots of each", text)
        self.assertIn("grid: decode geometric mean", text)
        self.assertIn("floors (tp4-ring, tiny): grid default 0.43%; the recorded defaults", text)
        self.assertNotIn("reference's (ref-again)", plain)


class StartupMemoryStartTest(unittest.TestCase):
    @staticmethod
    def series(rows: list[tuple[float, float, float, float]]) -> str:
        kib = lambda gib: int(gib * 1024 ** 2)
        return "".join(f"{t} {kib(a)} {kib(f)} {kib(c)}\n" for t, a, f, c in rows)

    def test_the_old_engine_is_left_out(self):
        text = self.series([(0.0, 5.4, 3.0, 4.0), (0.25, 5.3, 2.9, 4.0),  # the old engine still serving
                            (0.5, 60.0, 50.0, 4.0), (0.75, 116.2, 110.0, 3.7),  # it stopped
                            (1.0, 40.0, 30.0, 3.8), (1.25, 6.0, 3.1, 4.8), (1.5, 6.4, 3.3, 4.7)])
        summary = lab.memory_series(text)
        self.assertEqual(summary["samples"], 7)
        self.assertEqual(summary["old_engine_stopped_at_seconds"], 0.8)
        self.assertEqual(summary["available_after_old_engine_stopped_gib"], 116.2)
        self.assertEqual((summary["min_available_gib"], summary["min_available_at_seconds"]), (6.0, 1.2))
        self.assertEqual(summary["min_free_gib"], 3.1)
        self.assertEqual(summary["before_old_engine_stopped"], {"samples": 3, "min_available_gib": 5.3,
                                                                "min_free_gib": 2.9})
        self.assertEqual(summary["first"]["available_gib"], 5.4)  # the raw first sample stays reported

    def test_no_old_engine(self):
        summary = lab.memory_series(self.series([(0.0, 117.0, 110.0, 1.0), (0.5, 20.0, 10.0, 2.0)]))
        self.assertEqual((summary["old_engine_stopped_at_seconds"], summary["min_available_gib"]), (0.0, 20.0))
        self.assertNotIn("before_old_engine_stopped", summary)

    def test_a_failed_boot_that_frees_its_memory_starts_at_the_plateau(self):
        text = self.series([(0.0, 5.0, 3.0, 4.0), (0.5, 115.8, 110.0, 4.0), (1.0, 6.5, 3.0, 4.0),
                            (1.5, 117.1, 112.0, 4.0)])
        summary = lab.memory_series(text)
        self.assertEqual((summary["old_engine_stopped_at_seconds"], summary["min_available_gib"]), (0.5, 6.5))


class SuiteJobReferenceRepeatTest(unittest.TestCase):
    def test_a_named_repeat_reaches_every_table(self):
        spec = {"experiment": "experiments/x", "run": "r"}
        job = {"kind": "measure", "suite": ["screen", "single-stream"], "reference": "refs/c54-screen",
               "reference_repeat": "refs/c54-screen-second", "arms": [{"config": "a.json", "label": "c"},
                                                                      {"config": "a.json", "label": "c-again"}]}
        tables = [s for s in lab.suite_steps(spec, job, job["arms"]) if s["kind"] == "table"]
        self.assertEqual(tables[0]["argv"], ["scripts/servebench_suite_table.py", "--reference-repeat",
                                             "refs/c54-screen-second", "r", "c", "refs/c54-screen", "c-again"])
        self.assertEqual(tables[1]["argv"][1:3], ["--reference-repeat", "refs/c54-screen-second-single-stream"])
        self.assertEqual(tables[1]["reference_repeat"], "refs/c54-screen-second-single-stream")
        job.pop("reference_repeat")
        plain = [s for s in lab.suite_steps(spec, job, job["arms"]) if s["kind"] == "table"]
        self.assertEqual(plain[0]["argv"], ["scripts/servebench_suite_table.py", "r", "c", "refs/c54-screen", "c-again"])

    def test_reference_problems_check_a_named_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, digest in (("ref", suite.digest(suite.load("screen"))), ("ref-other", "e" * 64)):
                (root / name).mkdir()
                (root / name / "suite.json").write_text(json.dumps({"suite_sha256": digest}))
            step = {"unresolved": None, "arms": 2, "reference": "ref", "suite": "screen", "reference_repeat": "ref-other"}
            original = lab.ROOT
            lab.ROOT = root
            try:
                self.assertEqual(lab.reference_problems(step), ["reference repeat ref-other is not a run of the reference's suite"])
                step["reference_repeat"] = "missing"
                self.assertEqual(lab.reference_problems(step), ["reference repeat missing is not a suite run (no suite.json)"])
            finally:
                lab.ROOT = original


class FirstBootTest(unittest.TestCase):
    cluster = {"host": {"home": "/home/u"}, "deployment": {"path": "{home}/projects/spark-ds41f"},
               "container": {"mounts": [["{checkout}/cache", "/cache", "rw"]], "image": "spark-ds41f:x"},
               "environment": {"TILELANG_CACHE_DIR": "/cache/spark-ds41f/jit/tilelang",
                               "TRITON_CACHE_DIR": "/cache/spark-ds41f/jit/triton",
                               "VLLM_CACHE_ROOT": "/cache/spark-ds41f/jit/vllm", "HF_HOME": "/cache/huggingface"}}

    def test_cache_directories_map_through_the_mount(self):
        self.assertEqual(lab.jit_cache_dirs(self.cluster),
                         [f"/home/u/projects/spark-ds41f/cache/spark-ds41f/jit/{d}" for d in ("tilelang", "triton", "vllm")])

    def test_probe_output(self):
        found = lab.parse_boot_probe("image sha256:abc\njit 273 /c/tilelang\njit 0 /c/triton\n")
        self.assertEqual(found, {"image_id": "sha256:abc", "jit_new_files": {"tilelang": 273, "triton": 0}})
        self.assertEqual(lab.parse_boot_probe("image -\n")["image_id"], None)

    def test_ledger_marks_an_image_first_boot_per_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "boot-ledger.jsonl"
            first = lab.mark_first_boots({"dgx1": {"image_id": "sha256:a", "jit_new_files": {"tilelang": 273}},
                                          "dgx2": {"image_id": "sha256:a", "jit_new_files": {"tilelang": 270}}},
                                         "tp4.json", 136.4, ledger)
            self.assertEqual([s["first_boot"] for s in first.values()], [True, True])
            again = lab.mark_first_boots({"dgx1": {"image_id": "sha256:a", "jit_new_files": {"tilelang": 0}},
                                          "dgx3": {"image_id": "sha256:a", "jit_new_files": {"tilelang": 0}}},
                                         "tp4-dev.json", 112.0, ledger)
            self.assertEqual((again["dgx1"]["first_boot"], again["dgx3"]["first_boot"]), (False, True))
            self.assertEqual(len(ledger.read_text().splitlines()), 2)
            self.assertEqual(lab.mark_first_boots({}, "x.json", None, ledger), {})
            self.assertEqual(len(ledger.read_text().splitlines()), 2)
        line = lab.describe_boot_probe("experiments/x/tp4.json", first)
        self.assertEqual(line, "boot tp4.json: first boot of the image on dgx1, dgx2; kernel cache dgx1 +273, dgx2 +270 "
                               "files (kernels compiled during the boot: leave its time to ready out of comparisons)")
        warm = lab.describe_boot_probe("tp4-dev.json", {"dgx1": {"first_boot": False, "jit_new_files": {"tilelang": 0}}})
        self.assertEqual(warm, "boot tp4-dev.json: not the image's first boot on any node; kernel cache dgx1 +0 files")

    def test_verdicts_leave_first_boots_out_of_time_to_ready(self):
        log = [
            "2026-11-10T13:27:48Z boot tp4.json: ready +136.4 s, command 144 s",
            "2026-11-10T13:27:50Z boot tp4.json: first boot of the image on dgx1, dgx2; kernel cache dgx1 +273, "
            "dgx2 +270 files (kernels compiled during the boot: leave its time to ready out of comparisons)",
            "2026-11-10T13:29:59Z token_determinism_text.py c exit 0",
            "2026-11-10T13:36:40Z boot tp4-dev.json: ready +113.4 s, command 121 s",
            "2026-11-10T13:36:42Z boot tp4-dev.json: not the image's first boot on any node; kernel cache dgx1 +0, dgx2 +0 files",
            "2026-11-10T13:48:25Z bench c exit 0",
            "2026-11-10T13:50:24Z boot tp4-dev.json: ready +112.1 s, command 119 s",
            "2026-11-10T14:02:08Z bench c-again exit 0",
        ]
        arms = candidate.headline_numbers(log, ["c", "c-again"])
        self.assertEqual(arms["c"].boots, [("tp4.json", 136.4, 144, False), ("tp4-dev.json", 113.4, 121, False)])
        self.assertEqual(arms["c"].probes, [(["dgx1", "dgx2"], {"dgx1": 273, "dgx2": 270}), ([], {"dgx1": 0, "dgx2": 0})])
        self.assertEqual(arms["c-again"].probes, [None])
        described = "\n".join(candidate.describe_arm(arms["c"], "r"))
        self.assertIn("first boot of the image on dgx1, dgx2, kernel cache +543 files: left out of time-to-ready", described)
        self.assertIn("warm (no kernel compiled)", described)
        lines = candidate.ready_comparison(arms)
        self.assertIn("  c: 113.4 s over 1 warm boot", lines)
        self.assertIn("  c-again: 112.1 s over 1 warm boot", lines)
        self.assertIn("  left out: c (tp4.json, +136.4 s)", lines)
        self.assertIn("  1 boot without a kernel-cache record (a log before the probe) counted as warm", lines)
        # A ledger that starts after an image's first boot marks its next boot first; with no
        # kernel compiled it is warm all the same.
        self.assertFalse(candidate.cold_boot((["dgx1"], {"dgx1": 0})))
        self.assertTrue(candidate.cold_boot((["dgx1"], {})))
        self.assertTrue(candidate.cold_boot(([], {"dgx1": 3})))


if __name__ == "__main__":
    unittest.main()
