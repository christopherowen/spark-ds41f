"""Source-only CPU mocks: exact observer scope, process binding and owned cleanup.

No test in this file connects to a node, opens host /proc, or starts Docker.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observer = load("lossless_observer_scope_test_support", HERE / "engram/observer_support.py")
scope = load("lossless_observer_scope_test_window", HERE / "scoped_window.py")
IMAGE = "sha256:" + "a" * 64
CID = "b" * 64
WINDOW = "dgx1-exact-captured-token"


def target():
    return {"container": "serving", "container_id": CID, "image": IMAGE,
            "init_pid": 42, "started_at": "first-boot", "processes": {
                "42": {"start_ticks": 100, "namespace_pids": [42, 1]},
                "73": {"start_ticks": 200, "namespace_pids": [73, 9]}}}


def hold():
    return {"holder": scope.HOLDER, "window": WINDOW, "nodes": sorted(scope.NAMES),
            "production_config": scope.RESTORE, "resource_scope": {
                "holder": scope.HOLDER, "kind": "nodes", "token": WINDOW,
                "nodes": sorted(scope.NAMES), "restore": {
                    "nodes": sorted(scope.NAMES), "profile": scope.RESTORE,
                    "commit": "c" * 40,
                    "image": "sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11"}}}


def nodes():
    return {"ssh_user": "swank", "nodes": [
        {"name": "dgx" + str(i), "rank": i - 1, "head": i == 1,
         "management_ip": "192.0.2." + str(i)} for i in (1, 2, 3)]}


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.current = hold()
        self.writes = []
        self.lab = SimpleNamespace(HOLD="hold-path", read_hold=lambda: copy.deepcopy(self.current),
            write_json_atomic=lambda path, value: self.writes.append((path, copy.deepcopy(value))))
        self.spark = SimpleNamespace(configuration=lambda args: ({}, nodes(), {}))
        self.snapshots = [{"node": "dgx1", "exit": 0}]

    def check(self, actual=None, token=WINDOW):
        with patch.object(scope, "node_hold_observations", return_value=(self.snapshots, [])):
            return scope.observer_scope(self.lab, self.spark, actual or nodes(), token)

    def test_exact_lease_and_resolved_group_pass_without_mutation(self):
        self.assertEqual(self.check(), self.snapshots)
        self.assertEqual(self.writes, [])

    def test_missing_unscoped_or_wrong_token_refuses_before_node_inspection(self):
        for change in ("missing", "unscoped", "holder", "token", "restore"):
            self.current = hold()
            if change == "missing":
                self.current = None
            elif change == "unscoped":
                self.current.pop("resource_scope")
            elif change == "holder":
                self.current["holder"] = "another holder"
            elif change == "token":
                self.current["resource_scope"]["token"] = "another window"
            else:
                self.current["resource_scope"]["restore"]["commit"] = ""
            with self.subTest(change=change), patch.object(scope, "node_hold_observations") as read:
                with self.assertRaises(RuntimeError):
                    scope.observer_scope(self.lab, self.spark, nodes(), WINDOW)
                read.assert_not_called()
        self.current = hold()
        with self.assertRaises(RuntimeError):
            self.check(token="")

    def test_three_nodes_are_insufficient_when_resolved_group_or_user_differs(self):
        for change in ("name", "management_ip", "rank", "head", "ssh_user"):
            actual = nodes()
            if change == "ssh_user":
                actual[change] = "other"
            else:
                actual["nodes"][0][change] = "other"
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.check(actual)

    def test_foreign_or_unreadable_participating_hold_refuses(self):
        with patch.object(scope, "node_hold_observations", return_value=([], ["dgx2: foreign hold"])):
            with self.assertRaisesRegex(RuntimeError, "foreign hold"):
                scope.observer_scope(self.lab, self.spark, nodes(), WINDOW)

    def test_unknown_cleanup_latches_only_exact_owned_scope(self):
        before = hold()
        scope.observer_review(self.lab, WINDOW, "unreachable observer cleanup")
        self.assertEqual(len(self.writes), 1)
        _, after = self.writes[0]
        self.assertEqual(after["holder"], "lossless-memory-observer-review")
        self.assertEqual(after["resource_scope"]["token"], WINDOW)
        self.assertEqual(after["resource_scope"]["restore"], before["resource_scope"]["restore"])
        self.assertEqual(after["resource_scope"]["nodes"], before["resource_scope"]["nodes"])
        self.assertFalse(scope.owns(after, WINDOW))
        self.current = {"holder": "foreign", "window": "foreign"}
        scope.observer_review(self.lab, WINDOW, "must not overwrite a foreign hold")
        self.assertEqual(len(self.writes), 1)


class TargetTests(unittest.TestCase):
    def metadata(self, value=None):
        value = copy.deepcopy(value or target())
        calls = []
        def run(*argv):
            calls.append(argv)
            if argv[:2] == ("docker", "inspect"):
                return json.dumps({"Id": value["container_id"], "Image": value["image"],
                    "State": {"Running": True, "Pid": value["init_pid"], "StartedAt": value["started_at"]},
                    "User": "0", "Privileged": False, "CapAdd": ["IPC_LOCK"]})
            if argv[:2] == ("docker", "top"):
                return "PID\n" + "\n".join(value["processes"]) + "\n"
            return json.dumps(value["processes"])
        return run, calls

    def test_capture_top_is_bound_to_inspected_immutable_id(self):
        run, calls = self.metadata()
        actual = observer.capture_target(run, "serving", lambda text: text.split()[1:], IMAGE)
        self.assertEqual(actual, target())
        self.assertEqual(next(row for row in calls if row[:2] == ("docker", "top"))[2], CID)
        self.assertEqual(observer.namespace_target(actual), {"container_id": CID, "init_pid": 1,
            "processes": {"1": {"start_ticks": 100}, "9": {"start_ticks": 200}}})

    def test_post_observation_rejects_birth_boot_image_and_pid_set_changes(self):
        for change in ("birth", "boot", "image", "pid", "container"):
            changed = target()
            if change == "birth":
                changed["processes"]["73"]["start_ticks"] += 1
            elif change == "boot":
                changed["started_at"] = "repeat-boot"
            elif change == "image":
                changed["image"] = "sha256:" + "c" * 64
            elif change == "container":
                changed["container_id"] = "d" * 64
            else:
                changed["processes"].pop("73")
            run, _ = self.metadata(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                observer.verify_target(run, target(), lambda text: text.split()[1:])

    def test_proc_contract_rejects_birth_namespace_and_cgroup_changes(self):
        namespace = {}
        exec(observer.PROC_IDENTITY, namespace)
        init = {"start_ticks": 100, "pid_namespace": "pid:[101]", "cgroup": ["0::/docker/" + CID]}
        namespace["observer_check_identity"](copy.deepcopy(init), 100, init, CID)
        for field, bad in (("start_ticks", 101), ("pid_namespace", "pid:[102]"),
                           ("cgroup", ["0::/docker/" + "d" * 64])):
            changed = {**init, field: bad}
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "PID reuse"):
                namespace["observer_check_identity"](changed, 100, init, CID)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                namespace["observer_assert_stable"](init, changed)
        unrelated = {**init, "cgroup": ["0::/docker/other"]}
        with self.assertRaises(RuntimeError):
            namespace["observer_check_identity"](unrelated, 100, unrelated, CID)


class FakeDocker:
    """In-memory daemon response script; never dispatches a subprocess."""
    def __init__(self, *, failure=None, foreign=None, cleanup_unreadable=False):
        self.calls, self.actual = [], None
        self.failure, self.foreign, self.cleanup_unreadable = failure, foreign, cleanup_unreadable
        self.started = False

    def __call__(self, argv, *, input_text=None, timeout=30):
        self.calls.append({"argv": list(argv), "timeout": timeout, "input_text": input_text})
        action = argv[1]
        if action == "create":
            labels = {argv[i + 1].split("=", 1)[0]: argv[i + 1].split("=", 1)[1]
                      for i, value in enumerate(argv[:-1]) if value == "--label"}
            self.actual = {"Id": CID, "Name": "/" + argv[argv.index("--name") + 1],
                           "Image": IMAGE, "ConfigImage": IMAGE, "Labels": labels, "Running": False}
            if self.failure == "create-timeout":
                raise subprocess.TimeoutExpired(argv, timeout, output=b"create reply lost")
            return SimpleNamespace(returncode=0, stdout=CID + "\n", stderr="")
        if action == "inspect":
            if self.cleanup_unreadable and self.started:
                return SimpleNamespace(returncode=255, stdout="", stderr="SSH connection lost")
            if self.actual is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object: " + argv[2])
            value = copy.deepcopy(self.actual)
            if self.foreign == "image":
                value["Image"] = "sha256:" + "d" * 64
            elif self.foreign == "label":
                value["Labels"][observer.LABEL_PREFIX + "nonce"] = "foreign"
            elif self.foreign == "id":
                value["Id"] = "e" * 64
            return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")
        if action == "start":
            self.started = True
            self.actual["Running"] = True
            if self.failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout, output=b"partial output", stderr=b"partial stderr")
            self.actual["Running"] = False
            return SimpleNamespace(returncode=7 if self.failure == "exit" else 0,
                                   stdout="native verdict\n", stderr="native error" if self.failure else "")
        if action == "stop":
            self.actual["Running"] = False
            return SimpleNamespace(returncode=0, stdout=CID + "\n", stderr="")
        if action == "rm":
            self.actual = None
            return SimpleNamespace(returncode=0, stdout=CID + "\n", stderr="")
        raise AssertionError("unexpected mocked Docker operation")


class LifecycleTests(unittest.TestCase):
    def invoke(self, docker, *, guard=lambda: None, trace=None, review=None, saved=None, timeout=30):
        trace = {} if trace is None else trace
        argv = ["docker", "run", "--rm", "-i", "--runtime=runc", "--cpus=1",
                "--memory=128m", "--memory-swap=128m", "--network=none", "--read-only",
                "--env", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "python3", IMAGE,
                "-c", "print('fixed read-only observer')"]
        return observer.owned_observer(docker, argv, image=IMAGE, window=WINDOW,
            guard=guard, latch_review=(review if review is not None else []).append, trace=trace,
            nonce="c" * 32, timeout=timeout, input_text="published verifier source",
            persist=(lambda: saved.append(copy.deepcopy(trace))) if saved is not None else None)

    def test_normal_cleanup_and_cid_are_persisted_before_start(self):
        docker, trace, saved, review = FakeDocker(), {}, [], []
        result = self.invoke(docker, trace=trace, saved=saved, review=review)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(trace["cleanup_known"])
        self.assertEqual(review, [])
        self.assertIsNone(docker.actual)
        create = docker.calls[0]["argv"]
        self.assertNotIn("--rm", create)
        self.assertIn("signal.alarm(30)", create[create.index("-c") + 1])
        self.assertTrue(any(row.get("cid") == CID and not any(
            c["argv"][1] == "start" for c in row["commands"]) for row in saved))
        self.assertEqual([row["argv"][-1] for row in docker.calls if row["argv"][1] in ("start", "rm")], [CID, CID])
        self.assertTrue(all(row["timeout"] <= 30 for row in docker.calls))

    def test_ssh_timeout_retains_partial_output_and_stops_only_owned_cid(self):
        docker, trace, review = FakeDocker(failure="timeout"), {}, []
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke(docker, trace=trace, review=review)
        self.assertEqual(review, [])
        self.assertTrue(trace["cleanup_known"])
        self.assertIsNone(docker.actual)
        self.assertIn("TimeoutExpired", trace["failure"])
        started = next(row for row in trace["commands"] if row["argv"][1] == "start")
        self.assertEqual(started["stdout"], "partial output")
        self.assertEqual(started["stderr"], "partial stderr")
        self.assertEqual([row["argv"][-1] for row in docker.calls if row["argv"][1] in ("stop", "rm")], [CID, CID])

    def test_lost_create_reply_recovers_only_exact_name_and_labels(self):
        docker, trace = FakeDocker(failure="create-timeout"), {}
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke(docker, trace=trace)
        self.assertTrue(trace["cleanup_known"])
        self.assertIsNone(docker.actual)
        self.assertFalse(any(row["argv"][1] == "start" for row in docker.calls))

    def test_foreign_image_labels_or_id_never_stop_or_remove(self):
        for foreign in ("image", "label", "id"):
            docker, trace, review = FakeDocker(foreign=foreign), {}, []
            with self.subTest(foreign=foreign), self.assertRaises(observer.CleanupUnknown):
                self.invoke(docker, trace=trace, review=review)
            self.assertFalse(any(row["argv"][1] in ("start", "stop", "rm") for row in docker.calls))
            self.assertFalse(trace["cleanup_known"])
            self.assertEqual(len(review), 1)

    def test_unreachable_cleanup_keeps_original_timeout_and_latches_review(self):
        docker = FakeDocker(failure="timeout", cleanup_unreadable=True)
        trace, review = {}, []
        with self.assertRaises(observer.CleanupUnknown):
            self.invoke(docker, trace=trace, review=review)
        self.assertIn("TimeoutExpired", trace["failure"])
        self.assertIn("cleanup_error", trace)
        self.assertFalse(trace["cleanup_known"])
        self.assertEqual(len(review), 1)
        self.assertFalse(any(row["argv"][1] in ("stop", "rm") for row in docker.calls))

    def test_failed_scope_before_create_dispatches_nothing(self):
        docker, trace, review = FakeDocker(), {}, []
        def refuse():
            raise RuntimeError("foreign exact scope")
        with self.assertRaisesRegex(RuntimeError, "foreign exact scope"):
            self.invoke(docker, guard=refuse, trace=trace, review=review)
        self.assertEqual(docker.calls, [])
        self.assertEqual(review, [])
        self.assertTrue(trace["cleanup_known"])

    def test_scope_change_after_create_cleans_owned_container_without_start(self):
        docker, review = FakeDocker(), []
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("scope changed")
        with self.assertRaisesRegex(RuntimeError, "scope changed"):
            self.invoke(docker, guard=guard, review=review)
        self.assertIsNone(docker.actual)
        self.assertEqual(review, [])
        self.assertFalse(any(row["argv"][1] == "start" for row in docker.calls))

    def test_native_nonzero_exit_is_preserved_without_retry(self):
        docker, trace = FakeDocker(failure="exit"), {}
        result = self.invoke(docker, trace=trace)
        self.assertEqual(result.returncode, 7)
        self.assertTrue(trace["cleanup_known"])
        self.assertEqual(sum(row["argv"][1] == "start" for row in docker.calls), 1)

    def test_long_transport_does_not_extend_child_past_watchdog_margin(self):
        docker, trace = FakeDocker(failure="exit"), {}
        result = self.invoke(docker, trace=trace, timeout=1800)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(trace["transport_timeout_seconds"], 1800)
        self.assertEqual(trace["child_lifetime_seconds"], 600)
        self.assertLess(trace["child_lifetime_seconds"] + 30, 900)
        create = docker.calls[0]["argv"]
        self.assertIn("signal.alarm(600)", create[create.index("-c") + 1])
        self.assertEqual(next(row for row in docker.calls if row["argv"][1] == "start")["timeout"], 1800)
        self.assertTrue(trace["cleanup_known"])


if __name__ == "__main__":
    unittest.main()
