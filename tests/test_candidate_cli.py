"""Host-independent tests for bin/spark candidate (spark_ds41f/candidate.py), with a fake ssh."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from spark_ds41f import candidate  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "candidate"
SPEC = "experiments/2026-10-26-spark-ds41f-candidate/candidate-99/screen-tp4-switched.json"
PROFILE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-99/tp4-switched-dev.json"
SCREEN = {"experiment": "experiments/2026-10-26-spark-ds41f-candidate/candidate-99",
          "run": "candidate-99-screen-tp4-switched",
          "jobs": [{"kind": "sync"},
                   {"kind": "measure", "suite": "screen",
                    "arms": [{"config": "tp4-switched-dev.json", "label": "candidate-99"},
                             {"config": "tp4-switched-dev.json", "label": "candidate-99-again"}]}]}
CLEAN_PROFILE = {"notes": "candidate-99 at tp4-switched with development RPC for the engine grid",
                 "extends": {"config": "config/cluster.json", "shape": "tp4-switched"}}


def unwrap_script(argv: list[str]) -> tuple[str, str]:
    """(host, the script the tool asked for) from an ssh argument list."""
    assert argv[0] == "ssh" and "ConnectTimeout=10" in argv, argv
    host, remote = argv[-2], shlex.split(argv[-1])
    assert remote[:2] == ["bash", "-lc"], remote
    script = remote[2]
    if script.startswith("(\n"):
        script = script[2:script.index("\n)\n__rc=$?")]
    return host, script


class FakeRun:
    """subprocess.run for ssh: ``handler(host, script)`` returns (exit status, stdout), or None for
    an ssh failure (exit 255), or "cut" for output that ends before the sentinel."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def __call__(self, argv, **kwargs):
        host, script = unwrap_script(argv)
        self.calls.append((host, script))
        answer = self.handler(host, script)
        if answer is None:
            return subprocess.CompletedProcess(argv, 255, "", "ssh: connect to host: Connection timed out")
        if answer == "cut":
            return subprocess.CompletedProcess(argv, 0, "@@section done\n", "")
        code, out = answer
        return subprocess.CompletedProcess(argv, 0, f"{out}\n{candidate.END} {code}\n", "")

    def scripts(self) -> str:
        return "\n".join(script for _, script in self.calls)


def fake_ssh(handler, attempts: int = 3) -> tuple[candidate.Ssh, FakeRun]:
    run = FakeRun(handler)
    return candidate.Ssh(run=run, attempts=attempts, delay=0, sleep=lambda _: None), run


class FakeLocal:
    """The laptop checkout: files at origin/main from a dict."""

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.fetches = 0

    def fetch(self) -> None:
        self.fetches += 1

    def rev_parse(self, ref: str) -> str | None:
        return "a" * 40

    def reader(self, revision):
        return lambda path: self.files.get(path)


def state_output(*, hold: dict | None = None, pending=(), running=(), done=(), failed=(),
                 watchers=(), runs: dict | None = None) -> str:
    lines = ["Welcome to the node", "@@section hold"]
    if hold:
        lines += json.dumps(hold, indent=2).splitlines()
    lines += ["@@section head", "b" * 40]
    for name, entries in (("pending", pending), ("running", running), ("done", done), ("failed", failed)):
        lines.append(f"@@section {name}")
        lines += [f"{1794300000 + index} {entry}" for index, entry in enumerate(entries)]
    lines.append("@@section entries")
    for entry, run in (runs or {}).items():
        lines += [f"@@entry {entry}", f'  "run": "{run}",']
    lines += ["@@section watchers", *watchers, "@@section runners", "@@section containers",
              "spark-ds41f|spark-ds41f:candidate-99|Up 2 hours|experiments/x/tp4-switched-dev.json"]
    return "\n".join(lines)


def quiet(function, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        result = function(*args, **kwargs)
    return result, out.getvalue()


MUTATING = ("git checkout", "git fetch", "cluster sync", "queue add", "lab_idle_start", "window close",
            "build distribute", "docker load")


class SshTest(unittest.TestCase):
    def test_ssh_failure_is_retried_never_empty(self):
        answers = iter([None, "cut", (0, "@@section done\nx.json\n@@section failed")])
        ssh, run = fake_ssh(lambda host, script: next(answers), attempts=3)
        result = ssh.call("sp1", "true")
        self.assertEqual(len(run.calls), 3)
        self.assertIn("x.json", result.stdout)

    def test_exhausted_retries_raise(self):
        ssh, run = fake_ssh(lambda host, script: None, attempts=2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(candidate.RemoteError):
            ssh.call("sp1", "true")
        self.assertEqual(len(run.calls), 2)

    def test_mutating_call_is_not_retried(self):
        ssh, run = fake_ssh(lambda host, script: None, attempts=4)
        with self.assertRaises(candidate.RemoteError):
            ssh.call("sp1", "python3 scripts/lab.py queue add x.json", retry=False)
        self.assertEqual(len(run.calls), 1)

    def test_remote_command_is_one_quoted_bash_string(self):
        argv = candidate.Ssh.argv("dgx1", candidate.in_checkout(candidate.TESTBEDS["tp4-ring"],
                                                                 candidate.queue_add_command("a b.json")))
        self.assertEqual(argv[:2], ["ssh", "-o"])
        self.assertEqual(shlex.split(argv[-1])[:2], ["bash", "-lc"])
        self.assertIn("\"$HOME\"/projects/spark-ds41f", shlex.split(argv[-1])[2])
        self.assertIn("'a b.json'", shlex.split(argv[-1])[2])


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.local = FakeLocal({SPEC: json.dumps(SCREEN, indent=1) + "\n",
                                PROFILE: json.dumps(CLEAN_PROFILE) + "\n",
                                "config/cluster.json": (ROOT / "config/cluster.json").read_text()})
        self.testbed = candidate.TESTBEDS["tp4-switched"]

    def handler(self, state: str, node_files: dict[str, str | None]):
        def answer(host, script):
            if "@@section hold" in script:
                return 0, state
            if "@@file" in script or "@@missing" in script:
                return 0, "\n".join(f"@@missing {p}" if text is None else f"@@file {p}\n{text}\n@@eof"
                                    for p, text in node_files.items())
            return 0, "4" * 40
        return answer

    def queue(self, ssh, apply: bool):
        return candidate.queue_specs(ssh, self.local, self.testbed, [SPEC], apply=apply)

    def test_refuses_to_move_the_checkout_mid_job(self):
        state = state_output(hold={"holder": "spark-lab: queue"}, running=["20261110T131527-other.json"])
        for apply in (False, True):
            ssh, run = fake_ssh(self.handler(state, {SPEC: None}))
            with self.assertRaises(candidate.Refused) as refused:
                quiet(self.queue, ssh, apply)
            self.assertIn("retry when the queue is idle", str(refused.exception))
            for word in MUTATING:
                self.assertNotIn(word, run.scripts(), f"issued {word!r} while a job runs")

    def test_mid_job_spec_must_match_origin(self):
        state = state_output(hold={"holder": "spark-lab: queue"}, running=["20261110T131527-other.json"])
        ssh, run = fake_ssh(self.handler(state, {SPEC: '{"run": "older"}\n'}))
        with self.assertRaises(candidate.Refused) as refused:
            quiet(self.queue, ssh, True)
        self.assertIn("differs from origin/main", str(refused.exception))
        self.assertNotIn("queue add", run.scripts())

    def test_mid_job_queues_from_the_current_checkout(self):
        state = state_output(hold={"holder": "spark-lab: queue"}, running=["20261110T131527-other.json"])
        ssh, run = fake_ssh(self.handler(state, {SPEC: self.local.files[SPEC]}))
        code, out = quiet(self.queue, ssh, True)
        self.assertEqual(code, 0)
        self.assertIn("queue add", run.scripts())
        for word in ("git checkout", "cluster sync", "lab_idle_start"):
            self.assertNotIn(word, run.scripts())

    def test_pending_in_an_open_window_counts_as_busy(self):
        state = state_output(hold={"holder": "spark-lab: queue"}, pending=["20261110T131527-next.json"])
        ssh, run = fake_ssh(self.handler(state, {SPEC: None}))
        with self.assertRaises(candidate.Refused):
            quiet(self.queue, ssh, True)
        self.assertNotIn("cluster sync", run.scripts())

    def test_foreign_hold_refused(self):
        ssh, run = fake_ssh(self.handler(state_output(hold={"holder": "agent bench"}), {}))
        with self.assertRaises(candidate.Refused):
            quiet(self.queue, ssh, True)
        self.assertEqual(len(run.calls), 1)

    def test_idle_plan_changes_nothing(self):
        ssh, run = fake_ssh(self.handler(state_output(), {}))
        code, out = quiet(self.queue, ssh, False)
        self.assertEqual(code, 0)
        self.assertIn("cluster sync --apply", out)
        self.assertIn("lab_idle_start.sh 10", out)
        self.assertEqual(len(run.calls), 1)  # the state read only

    def test_idle_apply_syncs_then_checks_then_queues_then_watches(self):
        ssh, run = fake_ssh(self.handler(state_output(), {}))
        code, out = quiet(self.queue, ssh, True)
        self.assertEqual(code, 0)
        order = [next(i for i, (_, s) in enumerate(run.calls) if word in s)
                 for word in ("git checkout -q --detach origin/main", "cluster sync --apply",
                              "run --dry-run", "queue add", "lab_idle_start.sh")]
        self.assertEqual(order, sorted(order))
        self.assertIn("setsid -f", run.calls[order[-1]][1])

    def test_idle_apply_refuses_dry_run_problems(self):
        def answer(host, script):
            if "@@section hold" in script:
                return 0, state_output()
            if "--dry-run" in script:
                return 0, "run x: 3 steps\n  problem: arm configuration x.json does not exist"
            return 0, "4" * 40
        ssh, run = fake_ssh(answer)
        with self.assertRaises(candidate.Refused):
            quiet(self.queue, ssh, True)
        self.assertNotIn("queue add", run.scripts())

    def test_already_queued_run_is_not_added_twice(self):
        state = state_output(pending=["20261110T131527-screen-tp4-switched.json"],
                             runs={"20261110T131527-screen-tp4-switched.json": SCREEN["run"]})
        ssh, run = fake_ssh(self.handler(state, {}))
        code, out = quiet(self.queue, ssh, True)
        self.assertIn("already pending or running", out)
        self.assertNotIn("queue add", run.scripts())


class WaitTest(unittest.TestCase):
    def listing(self, done, failed=(), running=()):
        lines = ["@@section done", *done, "@@section failed", *failed]
        return 0, "\n".join(lines)

    def run_wait(self, answers, timeout=1000.0):
        ssh, run = fake_ssh(lambda host, script: next(answers), attempts=2)
        now = [0.0]
        args = argparse.Namespace(testbeds=["tp4-switched"], timeout=timeout, interval=10)

        def sleep(seconds):
            now[0] += seconds
        code, out = quiet(candidate.command_wait, args, ssh=ssh, clock=lambda: now[0], sleep=sleep)
        return code, out, run

    def test_sets_compare_done_and_failed_not_running(self):
        before = {"tp4-switched": (frozenset({"a.json"}), frozenset())}
        after = {"tp4-switched": (frozenset({"a.json"}), frozenset({"b.json"}))}
        self.assertEqual(candidate.finished_changes(before, after),
                         {"tp4-switched": {"done": [], "failed": ["b.json"]}})
        self.assertEqual(candidate.finished_changes(before, before), {})
        # A listing never reads running/: the script lists done and failed by name.
        self.assertNotIn("running", candidate.finished_script())

    def test_ssh_failure_is_no_change(self):
        answers = iter([
            self.listing(["a.json"]),           # baseline
            None, None,                         # a poll whose two attempts both fail
            "cut", self.listing(["a.json"]),    # cut short, then unchanged
            self.listing(["a.json", "b.json"]),  # b finished
            (0, state_output(done=["a.json", "b.json"])),
        ])
        code, out, run = self.run_wait(answers)
        self.assertEqual(code, 0)
        self.assertIn("tp4-switched: done   b.json", out)
        self.assertNotIn("a.json\n", out.split("waiting for")[1].split("tp4-switched (")[0])

    def test_timeout_exits_2(self):
        answers = iter([self.listing(["a.json"])] + [self.listing(["a.json"])] * 200)
        code, out, run = self.run_wait(answers, timeout=50)
        self.assertEqual(code, 2)


class GateTest(unittest.TestCase):
    def files(self, folder: str) -> dict[str, str]:
        return {path.name: path.read_text() for path in sorted((FIXTURES / "determinism" / folder).glob("*.jsonl"))}

    def test_identical_run_and_its_hashes(self):
        lines, failures = candidate.gate_report(self.files("candidate-54-tp4"),
                                                expect=("18d61961", "d6568649"),
                                                expect_text=("0a2dacdd", "12fd8257"))
        self.assertEqual(failures, [])
        text = "\n".join(lines)
        self.assertIn("determinism-candidate-54.jsonl: log-probability run; identical: yes", text)
        self.assertIn("log-probability hashes: reference 18d61961  long d6568649", text)
        self.assertIn("greedy-candidate-54.jsonl: text run (no log probabilities); identical: yes", text)
        self.assertIn("text hashes:            reference 0a2dacdd  long 12fd8257", text)

    def test_text_hashes_given_as_logprob_expectation_fail(self):
        # The past slip: comparing a text hash with a log-probability hash.
        lines, failures = candidate.gate_report(self.files("candidate-54-tp4"), expect=("0a2dacdd", "12fd8257"))
        self.assertTrue(any("log-probability hashes 18d61961/d6568649" in f for f in failures), failures)

    def test_scenario_diffs_name_prompts_and_positions(self):
        lines, failures = candidate.gate_report(self.files("candidate-55-978r2"))
        text = "\n".join(lines)
        self.assertIn("identical: NO", text)
        self.assertIn("again: 1 token diff(s), 1 log-probability diff(s)", text)
        self.assertIn("prompt 2: first token diff at 189, first log-probability diff at 176", text)
        self.assertNotIn("c8:", text)
        self.assertEqual(len(failures), 2)

    def test_shapes_differ_in_hashes(self):
        lines, failures = candidate.gate_report(self.files("candidate-54-tp3-switched"), expect=("18d61961", "d6568649"))
        self.assertTrue(failures)
        self.assertIn("reference 5c22ce41  long bce6092e", "\n".join(lines))

    def test_remote_files_round_trip(self):
        files = self.files("candidate-55-978r2")
        out = "\n".join(f"@@file {name}\n{text}\n@@eof" for name, text in files.items())
        self.assertEqual(candidate.parse_node_files(out), files)
        ssh, run = fake_ssh(lambda host, script: (0, out))
        args = argparse.Namespace(testbed="tp4-ring", run="candidate-55-validate-978r2-tp4", expect=None, expect_text=None)
        code, printed = quiet(candidate.command_gate, args, ssh=ssh)
        self.assertEqual(code, 1)
        self.assertIn("gate: FAIL", printed)
        self.assertIn("results/private/determinism/candidate-55-validate-978r2-tp4", run.calls[0][1])


class VerdictsTest(unittest.TestCase):
    text = (FIXTURES / "round2-window.txt").read_text()
    screen = "20261109T191158-candidate-51-screen-tp4.json"
    spec = {"run": "candidate-51-screen-tp4", "jobs": [{"kind": "measure", "suite": "screen",
            "arms": [{"label": "candidate-51"}, {"label": "candidate-51-again"}]}]}

    def test_segment_runs_from_started_to_done(self):
        segments = candidate.cut_segments(self.text, self.screen)
        self.assertEqual(len(segments), 1)
        self.assertIn("started", segments[0].lines[0])
        self.assertTrue(segments[0].lines[-1].endswith(f"QUEUE job {self.screen} done"))
        self.assertNotIn("candidate-51-candidate-tp4.json started", "\n".join(segments[0].lines))

    def test_verdict_block_ends_at_the_blank_line(self):
        lines = candidate.cut_segments(self.text, self.screen)[0].lines
        blocks = candidate.verdict_blocks(lines)
        self.assertEqual(len(blocks), 1)
        suite, block = blocks[0]
        self.assertEqual(suite, "screen")
        self.assertTrue(block[0].startswith("verdicts: candidate-50-no1009-tp4-ring-screen (recorded) -> candidate-51"))
        self.assertTrue(block[-1].startswith("paired rows: margin"))
        self.assertTrue(any(line.startswith("c16: decode tok/s, paired") for line in block))
        self.assertFalse(any(line.startswith("rows a step") for line in block))

    def test_headline_numbers(self):
        lines = candidate.cut_segments(self.text, self.screen)[0].lines
        arms = candidate.headline_numbers(lines, candidate.spec_labels(self.spec))
        self.assertEqual(list(arms), ["candidate-51", "candidate-51-again"])
        first, again = arms["candidate-51"], arms["candidate-51-again"]
        self.assertEqual(first.boots, [("tp4-dev.json", 112.0, 118, False)])
        self.assertEqual(again.boots, [("tp4-dev.json", 113.0, 119, False)])
        self.assertAlmostEqual(first.closed[("candidate-51-screen-tp4", 16)][0], 271.403, places=3)
        self.assertAlmostEqual(first.prefill[("candidate-51-screen-tp4", 32768)], 8660.59, places=2)
        self.assertAlmostEqual(first.model["candidate-51-screen-tp4"]["acceptance_length"], 2.3057, places=4)
        printed = "\n".join(candidate.describe_arm(first, "candidate-51-screen-tp4"))
        self.assertIn("closed loop c16: 271.40 tok/s", printed)
        self.assertIn("boot tp4-dev.json: ready +112.0 s", printed)

    def test_gate_job_numbers(self):
        entry = "20261109T191155-candidate-51-candidate-tp4.json"
        lines = candidate.cut_segments(self.text, entry)[0].lines
        arms = candidate.headline_numbers(lines, ["candidate-51"])
        self.assertEqual(arms["candidate-51"].boots, [("tp4.json", 155.2, 162, False)])
        self.assertEqual(arms["candidate-51"].distinct["c16-distinct"], (16, 274.0, 0.14))
        self.assertEqual(candidate.verdict_blocks(lines), [])

    def test_remote_awk_cut_matches(self):
        if not shutil.which("bash") or not shutil.which("awk"):
            self.skipTest("bash and awk are needed")
        with tempfile.TemporaryDirectory() as home:
            lab = Path(home) / candidate.LAB
            lab.mkdir(parents=True)
            (lab / "round2-window.log").write_text(self.text)
            (lab / "other.log").write_text("nothing here\n")
            script = candidate.segment_script(self.screen, body=True)
            out = subprocess.run(["bash", "-c", candidate.Ssh.wrap(script)], cwd=home, text=True,
                                 capture_output=True, timeout=30).stdout
            code, body = candidate.Ssh.unwrap(out)
            self.assertEqual(code, 0)
            segments = candidate.parse_segments(body)
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].log, "round2-window.log")
            self.assertEqual((segments[0].start, segments[0].end), (55, 415))
            self.assertEqual(segments[0].lines, candidate.cut_segments(self.text, self.screen)[0].lines)

    def test_entries_by_run_name_are_exact(self):
        if not shutil.which("bash"):
            self.skipTest("bash is needed")
        with tempfile.TemporaryDirectory() as home:
            queue = Path(home) / candidate.QUEUE
            (queue / "done").mkdir(parents=True)
            for name, run in (("20261110T000001-a.json", "candidate-58-screen-tp4"),
                              ("20261110T000002-b.json", "candidate-58-screen-tp4-switched")):
                (queue / "done" / name).write_text(json.dumps({"experiment": "e", "run": run, "jobs": []}, indent=2) + "\n")
            out = subprocess.run(["bash", "-c", candidate.Ssh.wrap(candidate.entries_script("candidate-58-screen-tp4"))],
                                 cwd=home, text=True, capture_output=True, timeout=30).stdout
            entries = candidate.parse_entries(candidate.Ssh.unwrap(out)[1])
            self.assertEqual([(e.state, e.name) for e in entries], [("done", "20261110T000001-a.json")])


class OverlayTest(unittest.TestCase):
    folder = "experiments/2026-10-26-spark-ds41f-candidate/candidate-58"

    def args(self, **changes):
        values = dict(folder=self.folder, name="no999", tree="dgx1:~/spark-lab/vllm", revert="c0ffee1",
                      profiles="tp4", allow_later=False, note=None, apply=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def handler(self, changes: list[str], later: str = ""):
        def answer(host, script):
            if "diff-tree" in script:
                return 0, "\n".join(["@@section parents", "c0ffee1 beef123", "@@section subject",
                                     "perf: something", "@@section changes", *changes])
            if "log --format" in script:
                return 0, later
            raise AssertionError(f"unexpected {script}")
        return answer

    def test_empty_file_list_refused(self):
        ssh, run = fake_ssh(self.handler(["M\ttests/kernels/test_x.py", "A\ttests/new_test.py"]))
        with self.assertRaises(candidate.Refused) as refused:
            quiet(candidate.command_overlay, self.args(), ssh=ssh)
        self.assertIn("nothing to overlay", str(refused.exception))
        self.assertNotIn("archive", run.scripts())

    def test_later_commit_touching_a_file_refused(self):
        later = "1234567890abcdef1234567890abcdef12345678 fix: a later patch\n"
        ssh, run = fake_ssh(self.handler(["M\tvllm/models/deepseek_v4_1/attention.py"], later))
        with self.assertRaises(candidate.Refused) as refused:
            quiet(candidate.command_overlay, self.args(), ssh=ssh)
        self.assertIn("--allow-later", str(refused.exception))
        code, out = quiet(candidate.command_overlay, self.args(allow_later=True), ssh=ssh)
        self.assertEqual(code, 0)
        self.assertIn("reverts those later changes as well", out)
        self.assertIn("plan only", out)
        self.assertFalse((ROOT / self.folder / "no999").exists())
        self.assertFalse((ROOT / self.folder / "tp4-no999.json").exists())

    def test_added_files_refused(self):
        ssh, run = fake_ssh(self.handler(["M\tvllm/a.py", "A\tvllm/b.py"]))
        with self.assertRaises(candidate.Refused):
            quiet(candidate.command_overlay, self.args(), ssh=ssh)

    def test_archive_names_explicit_files(self):
        command = candidate.archive_command("~/t", "c0ffee1", ["vllm/a.py", "vllm/b c.py"])
        self.assertTrue(command.endswith("-- vllm/a.py 'vllm/b c.py'"))
        self.assertIn("c0ffee1^", command)

    def test_profiles_extend_with_cache_and_ro_mounts(self):
        mounts = candidate.base_mounts(self.folder, "tp4")
        self.assertEqual(mounts, [["{checkout}/cache", "/cache", "rw"]])
        profiles = candidate.overlay_profiles(self.folder, "no999", "tp4", ["vllm/x/y.py"], mounts, False, None)
        main = profiles[f"{self.folder}/tp4-no999.json"]
        self.assertEqual(main["extends"], {"config": f"{self.folder}/tp4.json"})
        self.assertEqual(main["set"]["container"]["mounts"], [
            ["{checkout}/cache", "/cache", "rw"],
            [f"{{checkout}}/{self.folder}/no999/overlay/vllm/x/y.py",
             "/opt/spark-ds41f/candidate/vllm/vllm/x/y.py", "ro"]])
        dev = profiles[f"{self.folder}/tp4-no999-dev.json"]
        self.assertEqual(dev["set"], {"environment": {"VLLM_SERVER_DEV_MODE": "1"}})
        for path, data in profiles.items():
            text = candidate.checked_json(path, data)
            self.assertTrue(text.endswith("}\n"))
            self.assertEqual(candidate.cited_hashes(data["notes"]), [])


class FakeProcess:
    def __init__(self, data: bytes):
        self.stdout, self.stderr = io.BytesIO(data), io.BytesIO(b"")

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def tar_of(root: Path, names: list[str]) -> bytes:
    import tarfile
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in names:
            archive.add(root / name, arcname=name)
    return buffer.getvalue()


class ApplyTest(unittest.TestCase):
    """--apply paths against a temporary repository and a fake node: nothing leaves the test."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.node = Path(self.temp.name) / "node"
        (self.root / ".work").mkdir(parents=True)
        self.patch = unittest.mock.patch.object(candidate, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def write(self, base: Path, path: str, text: str) -> None:
        (base / path).parent.mkdir(parents=True, exist_ok=True)
        (base / path).write_text(text)

    def test_receipts_copy_and_never_overwrite(self):
        run, entry = "candidate-99-screen-tp4-switched", "20261110T131527-screen-tp4-switched.json"
        results = self.node / "results" / "private"
        for path, text in {f"lab/{run}-table.txt": "table\n", f"lab/overlays-{run}.txt": "# x\n",
                           f"determinism/{run}/startup-memory-candidate-99.json": "{}\n",
                           f"suites/{run}/candidate-99/suite.json": "{}\n",
                           f"suites/{run}/candidate-99/model.json.gz": "big",
                           f"suites/{run}/candidate-99/c16.engine-sp1.log": "log\n"}.items():
            self.write(results, path, text)
        spec = dict(SCREEN, run=run)
        copies = [f"lab/{run}-table.txt", f"lab/overlays-{run}.txt",
                  f"determinism/{run}/startup-memory-candidate-99.json", f"suites/{run}/candidate-99/suite.json"]

        def answer(host, script):
            if "@@entry" in script:
                return 0, f"@@entry done {entry}\n{json.dumps(spec, indent=2)}\n@@eof"
            if "find" in script:
                listing = [f"{(results / p).stat().st_size} {p}" for p in
                           sorted(str(f.relative_to(results)) for f in results.rglob("*") if f.is_file())
                           if not p.startswith("lab/queue")]
                return 0, "\n".join(listing + [f"@@missing lab/{run}"])
            if "awk" in script:
                return 0, (f"@@segment w-window.log 10\n2026 QUEUE job {entry} started\nverdicts: x\n"
                           f"2026 QUEUE job {entry} done\n@@segment-end 12")
            if "sha256sum" in script:
                return 0, f"{'ab' * 32}  suites/{run}/candidate-99/model.json.gz"
            raise AssertionError(script)

        ssh, run_calls = fake_ssh(answer)
        ssh.popen = lambda argv, **kw: FakeProcess(tar_of(results, copies))
        dest = "experiments/x/candidate-99/receipts/tp4-switched"
        args = argparse.Namespace(testbed="tp4-switched", reference=run, dest=dest, apply=True, force=False)
        code, out = quiet(candidate.command_receipts, args, ssh=ssh)
        self.assertEqual(code, 0, out)
        target = self.root / dest
        for path in copies:
            self.assertEqual((target / path).read_text(), (results / path).read_text())
        self.assertFalse((target / f"suites/{run}/candidate-99/model.json.gz").exists())
        self.assertFalse((target / f"suites/{run}/candidate-99/c16.engine-sp1.log").exists())
        self.assertIn(f"QUEUE job {entry} done", (target / f"lab/{run}-window.log").read_text())
        tsv = (target / "large-files-on-nodes.tsv").read_text()
        self.assertEqual(tsv, f"sha256\tbytes\tlocation\n{'ab' * 32}\t3\tsp1:~/projects/spark-ds41f/"
                              f"results/private/suites/{run}/candidate-99/model.json.gz\n")
        (target / copies[0]).write_text("edited\n")
        with self.assertRaises(candidate.Refused):
            quiet(candidate.command_receipts, args, ssh=ssh)
        self.assertEqual((target / copies[0]).read_text(), "edited\n")
        self.assertEqual(list((self.root / ".work").iterdir()), [])

    def test_overlay_writes_exactly_the_listed_files(self):
        folder = "experiments/x/candidate-99"
        for path in ("releases/spark-ds41f-1107/cluster.json",):
            self.write(self.root, path, (ROOT / path).read_text())
        self.write(self.root, f"{folder}/tp4.json", json.dumps(
            {"notes": "candidate-99 at tp4-ring", "extends": {"config": "releases/spark-ds41f-1107/cluster.json",
                                                              "shape": "tp4-ring"}}) + "\n")
        tree = self.node / "tree"
        self.write(tree, "vllm/models/a.py", "old a\n")
        changes = ["M\tvllm/models/a.py", "M\ttests/test_a.py"]

        def answer(host, script):
            if "diff-tree" in script:
                return 0, "\n".join(["@@section parents", "c0ffee1 beef123", "@@section subject", "perf: a",
                                     "@@section changes", *changes])
            if "log --format" in script:
                return 0, ""
            raise AssertionError(script)

        ssh, calls = fake_ssh(answer)
        streamed = []

        def popen(argv, **kw):
            streamed.append(unwrap_script(argv)[1])
            return FakeProcess(tar_of(tree, ["vllm/models/a.py"]))
        ssh.popen = popen
        args = argparse.Namespace(folder=folder, name="no999", tree="dgx1:~/t", revert="c0ffee1", profiles="tp4",
                                  allow_later=False, note=None, apply=True)
        code, out = quiet(candidate.command_overlay, args, ssh=ssh)
        self.assertEqual(code, 0, out)
        self.assertTrue(streamed[0].endswith("archive --format=tar 'c0ffee1^' -- vllm/models/a.py"), streamed)
        self.assertEqual((self.root / folder / "no999/overlay/vllm/models/a.py").read_text(), "old a\n")
        for name in ("tp4-no999.json", "tp4-no999-dev.json"):
            text = (self.root / folder / name).read_text()
            self.assertTrue(text.endswith("}\n") and not text.endswith("\n\n"))
            self.assertEqual(candidate.cited_hashes(json.loads(text)["notes"]), [])
        mounts = json.loads((self.root / folder / "tp4-no999.json").read_text())["set"]["container"]["mounts"]
        self.assertEqual(mounts[-1], [f"{{checkout}}/{folder}/no999/overlay/vllm/models/a.py",
                                      "/opt/spark-ds41f/candidate/vllm/vllm/models/a.py", "ro"])
        with self.assertRaises(candidate.Refused):  # a second run never overwrites
            quiet(candidate.command_overlay, args, ssh=ssh)

    def test_overlay_refuses_an_archive_of_more_files(self):
        folder = "experiments/x/candidate-99"
        self.write(self.root, "releases/spark-ds41f-1107/cluster.json",
                   (ROOT / "releases/spark-ds41f-1107/cluster.json").read_text())
        self.write(self.root, f"{folder}/tp4.json", json.dumps(
            {"extends": {"config": "releases/spark-ds41f-1107/cluster.json", "shape": "tp4-ring"}}) + "\n")
        tree = self.node / "tree"
        self.write(tree, "vllm/a.py", "a\n")
        self.write(tree, "vllm/b.py", "b\n")
        ssh, calls = fake_ssh(lambda host, script: (0, "@@section parents\nc0ffee1 beef123\n@@section changes\nM\tvllm/a.py")
                              if "diff-tree" in script else (0, ""))
        ssh.popen = lambda argv, **kw: FakeProcess(tar_of(tree, ["vllm/a.py", "vllm/b.py"]))
        args = argparse.Namespace(folder=folder, name="no999", tree="dgx1:~/t", revert="c0ffee1", profiles="tp4",
                                  allow_later=False, note=None, apply=True)
        with self.assertRaises(candidate.Refused):
            quiet(candidate.command_overlay, args, ssh=ssh)
        self.assertFalse((self.root / folder / "no999").exists())
        self.assertFalse((self.root / folder / "tp4-no999.json").exists())


class ShipTest(unittest.TestCase):
    def test_resumes_skipping_nodes_that_hold_the_image(self):
        image, wanted = "spark-ds41f:candidate-99", "sha256:" + "1" * 64
        held = {"dgx1", "sp1", "sp2", "sp3", "sp4"}  # tp4-switched already done; sp5..sp7 lack it
        folder = "experiments/2026-10-26-spark-ds41f-candidate/candidate-99"
        profile = {"extends": {"config": "config/cluster.json", "shape": "tp4-switched"},
                   "set": {"container": {"image": image}}}
        recipe_text = (ROOT / "config/cluster.json").read_text()
        local = FakeLocal({f"{folder}/tp4-switched.json": json.dumps(profile) + "\n",
                           f"{folder}/tp3-switched.json": json.dumps(dict(profile, extends={
                               "config": "config/cluster.json", "shape": "tp3-switched"})) + "\n",
                           "config/cluster.json": recipe_text})

        def answer(host, script):
            if script.startswith("docker image inspect"):
                return (0, wanted) if host in held else (1, "")
            if "docker save" in script and "| ssh" in script:
                held.add("sp5")
                return 0, "Loaded image: " + image
            if "@@section hold" in script:
                return 0, state_output()
            if "build distribute" in script:
                held.update({"sp6", "sp7"})
                return 0, "sp6: loaded"
            return 0, "5" * 40
        ssh, run = fake_ssh(answer)
        ssh.popen = lambda *a, **k: self.fail("streamed through the laptop although sp1 holds the image")
        args = argparse.Namespace(image=image, source="dgx1", to=["tp4-switched", "tp3-switched"], profile_dir=None,
                                  queue=None, idle_minutes=10, log_name=None, wait_timeout=10, stream_timeout=10,
                                  interval=1, attempts=2, apply=True)
        code, out = quiet(candidate.command_ship, args, ssh=ssh, local=local, sleep=lambda _: None)
        self.assertEqual(code, 0, out)
        relays = [(h, s) for h, s in run.calls if "docker save" in s]
        self.assertEqual([h for h, _ in relays], ["sp1"])
        self.assertIn("ssh -o ConnectTimeout=10", relays[0][1])
        distributes = [(h, s) for h, s in run.calls if "build distribute" in s]
        self.assertEqual([h for h, _ in distributes], ["sp5"])
        self.assertIn(f"{folder}/tp3-switched.json", distributes[0][1])
        self.assertIn("every node already holds", out)
        self.assertNotIn("DIFFERS", out)


class LintTest(unittest.TestCase):
    def read(self, files):
        return lambda path: files.get(path)

    def test_hash_in_notes(self):
        profile = {"notes": "candidate-99 at tp4-switched (tree 62df12dc)",
                   "extends": {"config": "config/cluster.json", "shape": "tp4-switched"}}
        files = {PROFILE: json.dumps(profile) + "\n",
                 "config/cluster.json": (ROOT / "config/cluster.json").read_text()}
        problems = candidate.lint_paths([PROFILE], self.read(files))
        self.assertEqual(len(problems), 1)
        self.assertIn("notes cite 62df12dc", problems[0])

    def test_own_tree_label_allowed(self):
        tree = "4815b7a03338917178850695f19aaabd0d1a78c6"
        profile = {"notes": "candidate-99 (vLLM 676 patches, tree 4815b7a03338)",
                   "extends": {"config": "config/cluster.json", "shape": "tp4-switched"},
                   "set": {"container": {"expected_labels": {"local.spark-ds41f.vllm.tree": tree}}}}
        files = {PROFILE: json.dumps(profile) + "\n",
                 "config/cluster.json": (ROOT / "config/cluster.json").read_text()}
        self.assertEqual(candidate.lint_paths([PROFILE], self.read(files)), [])

    def test_numbers_and_words_are_not_hashes(self):
        self.assertEqual(candidate.cited_hashes("1048576 tokens, 1000000, defaced, 0794 #956"), [])
        self.assertEqual(candidate.cited_hashes("tree ed1bdb7a7684;"), ["ed1bdb7a7684"])

    def test_trailing_newlines(self):
        files = {"a.md": "text\n\n", "b.md": "text", "c.md": "text\n", "d.json": "{}\n\n"}
        problems = candidate.lint_paths(list(files), self.read(files))
        self.assertEqual(problems, ["a.md: ends in more than one newline", "b.md: does not end in a newline",
                                    "d.json: ends in more than one newline"])

    def test_spec_references_missing_profile(self):
        files = {SPEC: json.dumps(SCREEN, indent=1) + "\n"}
        problems = candidate.lint_spec(self.read(files), SPEC)
        self.assertEqual(problems, [f"{SPEC}: references {PROFILE}, which does not exist"])

    def test_tool_written_json_is_checked(self):
        with self.assertRaises(candidate.Refused):
            candidate.checked_json("x.json", {"notes": "reverted from tree 62df12dc"})


class StatusTest(unittest.TestCase):
    def test_state_parse_and_description(self):
        hold = {"holder": "spark-lab: queue", "current": "boot x.json (candidate-99)", "heartbeat": "2026-11-10T14:56:42Z"}
        text = state_output(hold=hold, pending=["20261110T2-b.json"], running=["20261110T1-a.json"],
                            done=["20261110T0-z.json", "20261109T9-y.json"],
                            watchers=["3475261 bash scripts/lab_idle_start.sh 10 candidate-99-window"])
        state = candidate.parse_state(candidate.TESTBEDS["tp4-ring"], text)
        self.assertTrue(state.hold_is_queue and state.busy)
        self.assertEqual(state.running, ["20261110T1-a.json"])
        described = "\n".join(candidate.describe_state(state))
        self.assertIn("last done:    20261109T9-y.json", described)
        self.assertIn("idle watcher: running: 3475261", described)
        self.assertIn("container:    spark-ds41f spark-ds41f:candidate-99", described)

    def test_truncated_state_is_an_error(self):
        with self.assertRaises(candidate.RemoteError):
            candidate.parse_state(candidate.TESTBEDS["tp4-ring"], "@@section hold\n")

    def test_cli_wiring(self):
        from spark_ds41f import cli
        args = cli.parser().parse_args(["candidate", "gate", "tp4-ring", "candidate-58-tp4", "--expect", "8e46794a/11ec073c"])
        self.assertEqual((args.testbed, args.run, args.expect), ("tp4-ring", "candidate-58-tp4", "8e46794a/11ec073c"))
        args = cli.parser().parse_args(["candidate", "ship", "spark-ds41f:candidate-58", "--from", "dgx1",
                                        "--to", "tp4-switched", "tp3-switched"])
        self.assertFalse(args.apply)


if __name__ == "__main__":
    unittest.main()
