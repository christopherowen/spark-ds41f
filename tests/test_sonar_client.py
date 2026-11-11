"""The benchmark client: the Sonar pin, its checkout, the deployment's suites and the lab shims."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spark_ds41f import lab, sonar  # noqa: E402


class PinTest(unittest.TestCase):
    def test_the_pin_is_a_full_revision_of_the_sonar_repository(self):
        pin = sonar.pin()
        self.assertEqual(sonar.lock_problems(pin), [])
        self.assertEqual(pin["upstream"], "https://github.com/christopherowen/Sonar.git")
        self.assertEqual(pin["revision_url"], f"https://github.com/christopherowen/Sonar/commit/{pin['revision']}")

    def test_problems(self):
        self.assertEqual(sonar.lock_problems({"kind": "git", "upstream": "http://x", "revision": "abc"}),
                         ["upstream must be an https URL", "revision must be a full commit SHA",
                          "tracking_ref is not recorded"])

    def test_not_in_the_release_lock(self):
        """The release lock must equal the newest release's frozen one; the client is apart."""
        self.assertNotIn("sonar", json.loads((ROOT / "upstreams.lock.json").read_text())["sources"])


class CheckoutTest(unittest.TestCase):
    """The pinned commit, fetched once from the pin's URL (a local repository here) and verified."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.upstream = root / "upstream"
        (self.upstream / "sonar").mkdir(parents=True)
        (self.upstream / "sonar" / "__main__.py").write_text("print('sonar')\n")
        git = lambda *a: subprocess.run(["git", *a], cwd=self.upstream, check=True, capture_output=True, text=True)
        git("init", "-q")
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "one")
        self.revision = git("rev-parse", "HEAD").stdout.strip()
        git("config", "uploadpack.allowReachableSHA1InWant", "true")
        self.lock = root / "sonar.json"
        self.lock.write_text(json.dumps({"kind": "git", "upstream": "https://example.invalid/Sonar.git",
                                         "tracking_ref": "refs/heads/main", "revision": self.revision}))
        self.cache = root / "cache"

    def fetch(self):
        real = sonar._git

        def git(*args, cwd=None):  # the pin's URL, served by the local repository
            args = tuple(str(self.upstream) if a == "https://example.invalid/Sonar.git" else a for a in args)
            return real(*args, cwd=cwd)

        with mock.patch.object(sonar, "CACHE", self.cache), mock.patch.object(sonar, "_git", git), \
                mock.patch.object(sonar, "_VERIFIED", {}), mock.patch.dict(os.environ, {sonar.OVERRIDE_ENV: ""}):
            return sonar.checkout(self.lock)

    def test_fetched_once_and_verified(self):
        path = self.fetch()
        self.assertEqual(path, self.cache / self.revision)
        self.assertTrue((path / "sonar" / "__main__.py").is_file())
        self.assertEqual(self.fetch(), path)  # the second call finds it
        self.assertEqual([p.name for p in self.cache.iterdir()], [self.revision])  # no staging left

    def test_a_changed_checkout_is_refused(self):
        path = self.fetch()
        (path / "sonar" / "__main__.py").write_text("print('changed')\n")
        with self.assertRaisesRegex(sonar.SonarUnavailable, "not a clean checkout"):
            self.fetch()

    def test_a_revision_the_upstream_lacks(self):
        self.lock.write_text(json.dumps({"kind": "git", "upstream": "https://example.invalid/Sonar.git",
                                         "tracking_ref": "refs/heads/main", "revision": "0" * 40}))
        with self.assertRaises(sonar.SonarUnavailable):
            self.fetch()
        self.assertFalse(any(self.cache.iterdir()))


class SuitesTest(unittest.TestCase):
    def test_the_current_references_are_runs_of_the_deployment_suites(self):
        """The suites moved from servebench/suites/ unchanged: every reference a profile names
        is still a run of its suite, so a candidate compares with it."""
        suite = sonar.module("suite")
        recipe = json.loads((ROOT / "config/cluster.json").read_text())
        references = {entry.get("benchmark_reference") for entry in recipe["shapes"].values()}
        references = {r for r in references if r and (ROOT / r / "suite.json").is_file()}
        self.assertTrue(references)
        for reference in sorted(references):
            for companion in ("", "-single-stream", "-sampled"):
                index = json.loads((ROOT / (reference + companion) / "suite.json").read_text())
                with self.subTest(reference=reference + companion):
                    self.assertEqual(suite.digest(sonar.load_suite(index["suite"]["name"])), index["suite_sha256"])

    def test_names_resolve_to_the_deployment_files_first(self):
        self.assertEqual(sonar.suite_name("release"), str(ROOT / "config/suites/release.json"))
        self.assertEqual(sonar.suite_name("standard"), "standard")  # a Sonar built-in
        self.assertEqual(sonar.suite_arguments(["U", "--suite", "screen", "--out", "o"]),
                         ["U", "--suite", str(ROOT / "config/suites/screen.json"), "--out", "o"])
        self.assertEqual(sonar.suite_arguments(["U", "--suite=sampled"]),
                         ["U", "--suite=" + str(ROOT / "config/suites/sampled.json")])

    def test_every_deployment_suite_parses(self):
        main, suite = sonar.module("sonar.__main__"), sonar.module("suite")
        for path in sorted((ROOT / "config/suites").glob("*.json")):
            for step in suite.load(str(path))["steps"]:
                with self.subTest(suite=path.stem, step=step["id"]):
                    main.build_parser().parse_args(suite.arguments(step, "http://x", "o.json", "l", "c.json.gz"))


class ShimTest(unittest.TestCase):
    def capture(self, script: str, argv: list[str]) -> list[str]:
        """The Sonar command line a lab shim runs."""
        seen = []
        main = sonar.module("sonar.__main__")
        with mock.patch.object(main, "main", side_effect=lambda args: seen.append(args) or 0), \
                mock.patch.object(sys, "argv", [script, *argv]), mock.patch.dict(os.environ):
            namespace = {"__name__": "__main__", "__file__": str(ROOT / script)}
            with self.assertRaises(SystemExit) as code:
                exec(compile((ROOT / script).read_text(), script, "exec"), namespace)
            self.assertEqual(code.exception.code, 0)
        return seen[0]

    def test_every_shim_exists_and_names_its_command(self):
        for script, command in lab.SONAR_SHIMS.items():
            with self.subTest(script=script):
                self.assertTrue((ROOT / script).is_file())
                self.assertEqual(self.capture(script, ["http://h:8000"])[0], command)

    def test_the_probes_carry_the_reasoning_switch(self):
        off = json.dumps({"chat_template_kwargs": {"thinking": False}})
        for script in ("scripts/distinct_streams.py", "scripts/acceptance_sweep.py", "scripts/sonar_determinism.py",
                       "scripts/sonar_single.py", "scripts/sonar_ttft.py", "scripts/sonar_mixed.py"):
            with self.subTest(script=script):
                argv = self.capture(script, ["http://h:8000", "--tokens", "8"] if "ttft" not in script
                                    and "mixed" not in script else ["http://h:8000"])
                self.assertEqual(argv[argv.index("--extra-body") + 1], off)
        own = self.capture("scripts/distinct_streams.py", ["http://h", "--extra-body", "{}"])
        self.assertEqual(own.count("--extra-body"), 1)
        # Load, grid and model steps send what their suites and arguments say, as before.
        self.assertNotIn("--extra-body", self.capture("scripts/sonar_run.py", ["http://h", "--mode", "closed"]))

    def test_mixed_reads_the_long_prompts_it_always_read(self):
        argv = self.capture("scripts/sonar_mixed.py", ["http://h", "--rounds", "1"])
        self.assertEqual(argv[argv.index("--long-source") + 1], str(ROOT / "experiments/2026-09-29-determinism"))

    def test_the_suite_shim_runs_the_deployment_file(self):
        argv = self.capture("scripts/sonar_suite.py", ["http://h", "--suite", "single-stream", "--out", "o"])
        self.assertEqual(argv[argv.index("--suite") + 1], str(ROOT / "config/suites/single-stream.json"))

    def test_lab_profiles_use_only_sonar(self):
        for name, profile in lab.PROFILES.items():
            for script, _, _ in profile["extras"]:
                with self.subTest(profile=name, script=script):
                    self.assertIn(script, lab.SONAR_SHIMS)


class EnvironmentTest(unittest.TestCase):
    def test_the_stream_collector_follows_the_deployment_policy(self):
        for policy, collector in (("current", "current"), ("recorded-1030", "text-chunks")):
            env = sonar.environment({sonar.STREAM_POLICY_ENV: policy})
            self.assertEqual(env["SONAR_STREAM_COLLECTOR"], collector)
            self.assertTrue(env["PYTHONPATH"].startswith(str(sonar.checkout())))
        with self.assertRaises(ValueError):
            sonar.environment({sonar.STREAM_POLICY_ENV: "other"})

    def test_the_old_model_barrier_is_refused_not_ignored(self):
        with self.assertRaisesRegex(sonar.SonarUnavailable, "SONAR_MODEL_CONTRACT"):
            sonar.environment({sonar.LEGACY_CONTRACT_ENV: "/frozen.json"})
        self.assertIsNone(sonar.legacy_contract_problem({}))

    def test_bin_spark_sonar(self):
        self.assertEqual(sonar.run("--version", stdout=subprocess.DEVNULL), 0)
        printed = subprocess.run([sys.executable, str(ROOT / "bin/spark"), "sonar", "--path"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(Path(printed), sonar.checkout())


if __name__ == "__main__":
    unittest.main()
