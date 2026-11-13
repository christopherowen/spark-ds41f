"""Host-independent tests for `bin/spark cache status` and `cache clear`."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark

CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-55"
NODES3 = {"ssh_user": "u", "nodes": [
    {"name": "dgx1", "head": True, "rank": 0, "management_ip": "1", "roce_peer_hcas": {}},
    {"name": "dgx2", "head": False, "rank": 1, "management_ip": "2", "roce_peer_hcas": {}},
    {"name": "dgx3", "head": False, "rank": 2, "management_ip": "3", "roce_peer_hcas": {}}]}
JIT = "/cache/spark-ds41f/jit"
ENVIRONMENT = {
    "TILELANG_CACHE_DIR": f"{JIT}/tilelang", "CUDA_CACHE_PATH": f"{JIT}/nv-compute",
    "TRITON_CACHE_DIR": f"{JIT}/triton", "VLLM_CACHE_ROOT": f"{JIT}/vllm", "VLLM_CACHE_DIR": f"{JIT}/vllm",
    "XDG_CACHE_HOME": JIT, "HF_HOME": "/cache/huggingface",
}
HOST_JIT = "/h/r/cache/spark-ds41f/jit"


def cluster(environment: dict | None = None, mounts: list | None = None) -> dict:
    return {
        "deployment": {"path": "{home}/r"}, "host": {"home": "/h"},
        "container": {"name": "c", "image": "img", "mounts": mounts if mounts is not None else [
            ["{home}/hub", "/hub", "ro"], ["{checkout}/cache", "/cache", "rw"]]},
        "environment": dict(ENVIRONMENT if environment is None else environment),
    }


def completed(code: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr=stderr)


def survey(script: str, *, running: str = "", docker: int = 0, entries: bool = False) -> subprocess.CompletedProcess:
    """A node's answer to the survey script: every directory present with three files."""
    lines = [f"containers\t{docker}\t{running}"] if "docker ps" in script else []
    for index in range(script.count("\nd=")):
        lines.append(f"cache\t{index}\tpresent\t3\t3221225472\t1794036000\t1794136000")
    if entries:
        lines += ["entries\t0\tv1/linux-aarch64/kernels\t412", "entries\t0\tv1/cuda-binaries\t       7"]
    return completed(stdout="\n".join(lines) + "\n")


def removed(_script: str) -> subprocess.CompletedProcess:
    return completed(stdout="removed\t3\t2048\t1794036000\t1794136000\n")


class Run:
    """One command with its configuration and remote calls replaced."""

    def __init__(self, command, *, apply: bool | None = None, only: list[str] | None = None,
                 profile: dict | None = None, surveys: dict | None = None, clears: dict | None = None,
                 cluster_config: str | None = "config/cluster.json") -> None:
        self.calls: list[tuple[str, str, str]] = []
        surveys, clears, lock = surveys or {}, clears or {}, threading.Lock()

        def remote_shell(_nodes, node, script):
            kind = "clear" if "-delete" in script else "survey"
            with lock:
                self.calls.append((node["name"], kind, script))
            answer = (clears if kind == "clear" else surveys).get(node["name"], removed if kind == "clear" else survey)
            return answer(script)

        args = argparse.Namespace(cluster_config=cluster_config, shape=None)
        if apply is not None:
            args.apply, args.only = apply, only
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(spark, "configuration", return_value=(profile or cluster(), NODES3, {})), \
                mock.patch.object(spark, "remote_shell", side_effect=remote_shell), \
                mock.patch.object(spark, "run_ssh", side_effect=AssertionError("an SSH call outside remote_shell")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.code = command(args)
        self.out, self.err = out.getvalue(), err.getvalue()

    def scripts(self, kind: str) -> list[tuple[str, str]]:
        return sorted((node, script) for node, found, script in self.calls if found == kind)


class DirectoryMappingTest(unittest.TestCase):
    def directories(self, profile: dict) -> dict[str, str]:
        found, unset, refused = spark.profile_caches(profile, spark.JIT_CACHES)
        self.assertEqual((unset, refused), ([], []))
        return {directory["name"]: directory["host"] for directory in found}

    def test_the_promoted_recipe_maps_each_cache_through_the_cache_mount(self) -> None:
        profile = spark.read_cluster("config/cluster.json", "tp4-ring")
        mount = spark.host_path(profile, "{checkout}/cache")
        self.assertTrue(mount.endswith("/spark-ds41f/cache"))
        self.assertEqual(self.directories(profile), {
            "tilelang": f"{mount}/spark-ds41f/jit/tilelang", "nv-compute": f"{mount}/spark-ds41f/jit/nv-compute",
            "triton": f"{mount}/spark-ds41f/jit/triton", "vllm": f"{mount}/spark-ds41f/jit/vllm"})

    def test_a_candidate_profile_maps_the_same_way(self) -> None:
        profile = spark.read_cluster(f"{CANDIDATE}/tp4.json")
        mount = spark.host_path(profile, "{checkout}/cache")
        self.assertEqual(self.directories(profile)["tilelang"], f"{mount}/spark-ds41f/jit/tilelang")

    def test_an_arm_with_its_own_tilelang_directory_names_that_one(self) -> None:
        profile = spark.read_cluster(f"{CANDIDATE}/rows24-attribution/tp4-no982-dev.json")
        mount = spark.host_path(profile, "{checkout}/cache")
        found = self.directories(profile)
        self.assertEqual(found["tilelang"], f"{mount}/tilelang/c55-rows24")
        self.assertEqual(found["triton"], f"{mount}/spark-ds41f/jit/triton")

    def test_placeholders_are_rendered_as_the_mounts_are(self) -> None:
        self.assertEqual(self.directories(cluster())["nv-compute"], f"{HOST_JIT}/nv-compute")
        self.assertEqual(spark.cache_mount_source(cluster()), spark.expected_binds(cluster())[1].split(":")[0])

    def test_an_unset_variable_skips_its_cache_and_says_so(self) -> None:
        environment = {key: value for key, value in ENVIRONMENT.items() if key != "TRITON_CACHE_DIR"}
        found, unset, refused = spark.profile_caches(cluster({**environment, "CUDA_CACHE_PATH": ""}), spark.JIT_CACHES)
        self.assertEqual([directory["name"] for directory in found], ["tilelang", "vllm"])
        self.assertEqual(unset, ["nv-compute: the profile sets no CUDA_CACHE_PATH; skipped",
                                 "triton: the profile sets no TRITON_CACHE_DIR; skipped"])
        self.assertEqual(refused, [])

    def test_the_two_vllm_variables_name_one_directory_or_each_their_own(self) -> None:
        same, _, _ = spark.profile_caches(cluster(), {"vllm": spark.JIT_CACHES["vllm"]})
        self.assertEqual([(d["variables"], d["host"]) for d in same],
                         [(["VLLM_CACHE_ROOT", "VLLM_CACHE_DIR"], f"{HOST_JIT}/vllm")])
        apart, _, _ = spark.profile_caches(cluster({"VLLM_CACHE_ROOT": "/cache/a/root", "VLLM_CACHE_DIR": "/cache/a/dir"}),
                                           {"vllm": spark.JIT_CACHES["vllm"]})
        self.assertEqual([d["host"] for d in apart], ["/h/r/cache/a/root", "/h/r/cache/a/dir"])


class PathGuardTest(unittest.TestCase):
    def test_a_directory_two_below_the_mount_is_accepted(self) -> None:
        self.assertEqual(spark.cache_host_directory(cluster(), "/cache/tilelang/c55-rows24"),
                         "/h/r/cache/tilelang/c55-rows24")
        self.assertEqual(spark.cache_host_directory(cluster(), "/cache/a/./b/"), "/h/r/cache/a/b")

    def test_paths_a_clear_must_never_touch_are_refused(self) -> None:
        for path in ("/cache", "/cache/", "/cache/x", "/cache/huggingface", "/models/..", "/models/a/b",
                     "/cache/../etc/x", "/cache/a/b/../../../etc", "/cache/a/..", "cache/a/b", "a", ".", "/",
                     "//cache/a/b", "/cachex/a/b", " /cache/a/b", "~/cache/a/b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                spark.cache_host_directory(cluster(), path)

    def test_a_profile_without_exactly_one_cache_mount_is_refused(self) -> None:
        for mounts in ([], [["{home}/hub", "/hub", "ro"]],
                       [["{checkout}/cache", "/cache", "rw"], ["{home}/other", "/cache/", "rw"]],
                       [["/", "/cache", "rw"]], [["cache", "/cache", "rw"]], [["{home}/../x", "/cache", "rw"]]):
            with self.subTest(mounts=mounts), self.assertRaises(ValueError):
                spark.cache_host_directory(cluster(mounts=mounts), "/cache/a/b")

    def test_a_path_another_mount_covers_is_refused(self) -> None:
        mounts = [["{checkout}/cache", "/cache", "rw"], ["{home}/elsewhere", "/cache/a", "rw"]]
        with self.assertRaises(ValueError):
            spark.cache_host_directory(cluster(mounts=mounts), "/cache/a/b")
        self.assertEqual(spark.cache_host_directory(cluster(mounts=mounts), "/cache/b/c"), "/h/r/cache/b/c")

    def test_a_refused_variable_is_reported_and_an_unset_one_is_not_an_error(self) -> None:
        for value in ("/cache", "/cache/x", "/models/.."):
            with self.subTest(value=value):
                found, unset, refused = spark.profile_caches(
                    cluster({"TILELANG_CACHE_DIR": value}), {"tilelang": spark.JIT_CACHES["tilelang"]})
                self.assertEqual((found, unset), ([], []))
                self.assertEqual(len(refused), 1)
                self.assertIn(f"tilelang: TILELANG_CACHE_DIR={value}: ", refused[0])
        self.assertEqual(spark.profile_caches(cluster({}), {"tilelang": spark.JIT_CACHES["tilelang"]}),
                         ([], ["tilelang: the profile sets no TILELANG_CACHE_DIR; skipped"], []))

    def test_clear_refuses_a_profile_with_a_refused_variable_before_any_remote_call(self) -> None:
        for apply in (False, True):
            run = Run(spark.command_cache_clear, apply=apply, profile=cluster({**ENVIRONMENT, "TRITON_CACHE_DIR": "/cache"}))
            self.assertEqual((run.code, run.calls), (1, []))
            self.assertIn("ERROR: triton: TRITON_CACHE_DIR=/cache: ", run.err)

    def test_status_reports_a_refused_variable_and_still_shows_the_others(self) -> None:
        run = Run(spark.command_cache_status, profile=cluster({**ENVIRONMENT, "TRITON_CACHE_DIR": "/models/.."}))
        self.assertEqual(run.code, 1)
        self.assertIn("ERROR: triton: TRITON_CACHE_DIR=/models/..: ", run.err)
        self.assertIn(f"  tilelang: {HOST_JIT}/tilelang: ", run.out)
        self.assertNotIn("  triton: ", run.out)


class ScriptTest(unittest.TestCase):
    def directory(self, host: str) -> dict:
        return {"name": "tilelang", "variables": ["TILELANG_CACHE_DIR"], "container": "/cache/a/b", "host": host}

    def test_the_clear_script_deletes_only_relative_to_the_directory_it_entered(self) -> None:
        script = spark.cache_clear_script(cluster(), self.directory(f"{HOST_JIT}/tilelang"))
        lines = script.splitlines()
        self.assertEqual(lines[:3], ["set -u", f"dir={HOST_JIT}/tilelang", "root=/h/r/cache"])
        enter = lines.index('cd -- "$dir" || exit 4')
        deletes = [index for index, line in enumerate(lines) if "-delete" in line]
        self.assertEqual(len(deletes), 2)
        for index in deletes:
            self.assertGreater(index, enter)
            self.assertIn("find . -mindepth 1 -delete", lines[index])
        self.assertNotIn("rm ", script)
        self.assertNotIn("$dir/", script)
        self.assertLess(next(i for i, line in enumerate(lines) if "docker ps" in line), enter)

    def test_every_path_is_quoted(self) -> None:
        odd = "/h/r/cache/a/it's $(reboot); b"
        for script in (spark.cache_clear_script(cluster(), self.directory(odd)),
                       spark.cache_survey_script(cluster(), [self.directory(odd)], containers=True)):
            self.assertIn(f"={spark.shlex.quote(odd)}\n", script)
            self.assertEqual(script.count("reboot"), 1)

    def test_the_survey_script_is_read_only_and_lists_containers_only_for_a_clear(self) -> None:
        directories = spark.profile_caches(cluster(), spark.JIT_CACHES)[0]
        status = spark.cache_survey_script(cluster(), directories, containers=False)
        plan = spark.cache_survey_script(cluster(), directories, containers=True)
        for script in (status, plan):
            for word in ("-delete", "rm ", "sudo", "mkdir", "mv ", "tee"):
                self.assertNotIn(word, script)
        self.assertNotIn("docker", status)
        self.assertIn("docker ps --filter 'name=^c$'", plan)
        self.assertEqual(plan.count("linux-*/kernels"), 1)

    def test_the_summary_adds_sizes_and_keeps_the_oldest_and_newest_time(self) -> None:
        def summary(lines: str) -> str:
            return subprocess.run(["awk", spark.CACHE_SUMMARY_AWK], input=lines, text=True, check=True,
                                  stdout=subprocess.PIPE).stdout

        self.assertEqual(summary("100 1794036000.25\n50 1793936000.75\n4096 1794136000.1\n"),
                         "3\t4246\t1793936001\t1794136000")
        self.assertEqual(summary(""), "0\t0\t0\t0")

    def test_a_survey_is_parsed_and_an_incomplete_one_is_refused(self) -> None:
        text = ("Last login: noise\ncontainers\t0\tc=experiments/x/arm.json dsv41-old= \n"
                "cache\t0\tpresent\t2\t150\t1793936000\t1794036000\n"
                "entries\t0\tv1/linux-aarch64/kernels\t       5\ncache\t1\tabsent\n")
        found = spark.parse_cache_survey(text, 2)
        self.assertEqual(found["running"], [("c", "experiments/x/arm.json"), ("dsv41-old", None)])
        self.assertEqual(found["caches"], [
            {"present": True, "files": 2, "bytes": 150, "oldest": 1793936000, "newest": 1794036000,
             "entries": [("v1/linux-aarch64/kernels", 5)]},
            {"present": False, "entries": []}])
        self.assertEqual(spark.parse_cache_survey("containers\t1\tpermission denied \ncache\t0\tabsent\n", 1)["docker_error"],
                         "permission denied")
        for broken in ("cache\t0\tabsent\n", "cache\t0\tpresent\t\ncache\t1\tabsent\n", "cache\t7\tabsent\n"):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                spark.parse_cache_survey(broken, 2)

    def test_cache_text(self) -> None:
        self.assertEqual(spark.cache_text({"present": False}), "absent")
        self.assertEqual(spark.cache_text({"present": True, "files": 0, "bytes": 0, "oldest": 0, "newest": 0}),
                         "present, no files")
        self.assertEqual(
            spark.cache_text({"present": True, "files": 7033, "bytes": 1288490189, "oldest": 1794036000,
                              "newest": 1794499320}),
            "1.2 GiB in 7,033 files, modified 2026-11-07T07:20:00Z to 2026-11-12T16:02:00Z")
        self.assertEqual([spark.cache_size_text(size) for size in (0, 1023, 1024, 5 * 2**20)],
                         ["0 B", "1023 B", "1.0 KiB", "5.0 MiB"])


class ScriptOnDiskTest(unittest.TestCase):
    """The remote scripts under a real shell, on a temporary cache mount, with a stand-in docker."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.write_command("docker", "#!/bin/sh\nexit 0\n")
        self.write_command("sudo", f"#!/bin/sh\necho \"$@\" >> '{self.root}/sudo-calls'\nexit 1\n")
        self.cache = self.root / "mount" / "a" / "b"
        for relative in ("v1/linux-aarch64/kernels/k1/kernel.so", "v1/linux-aarch64/kernels/k2/kernel.so",
                         "v1/cuda-binaries/one.bin", ".hidden", "-rf", "name with space"):
            path = self.cache / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x" * 10)
        self.sibling = self.root / "mount" / "a" / "keep" / "file"
        self.sibling.parent.mkdir()
        self.sibling.write_text("keep")
        self.outside = self.root / "outside" / "file"
        self.outside.parent.mkdir()
        self.outside.write_text("keep")
        self.gnu_find = subprocess.run(["find", "--version"], capture_output=True, check=False).returncode == 0

    def write_command(self, name: str, text: str) -> None:
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def profile(self, **environment: str) -> dict:
        return cluster(environment, mounts=[[str(self.root / "mount"), "/cache", "rw"]])

    def shell(self, script: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", script], cwd=self.root / "outside", text=True, capture_output=True,
                              check=False, env={"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.root / "outside")})

    def clear(self, container_path: str) -> subprocess.CompletedProcess:
        profile = self.profile(TILELANG_CACHE_DIR=container_path)
        (directory,), _, refused = spark.profile_caches(profile, {"tilelang": spark.JIT_CACHES["tilelang"]})
        self.assertEqual(refused, [])
        return self.shell(spark.cache_clear_script(profile, directory))

    def assert_untouched(self) -> None:
        self.assertEqual(self.sibling.read_text(), "keep")
        self.assertEqual(self.outside.read_text(), "keep")

    def test_a_clear_empties_the_directory_and_keeps_it_and_its_neighbours(self) -> None:
        result = self.clear("/cache/a/b")
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertTrue(result.stdout.startswith("removed\t"))
        if self.gnu_find:
            self.assertTrue(result.stdout.startswith("removed\t6\t60\t"))
        self.assertTrue(self.cache.is_dir())
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assert_untouched()
        self.assertFalse((self.root / "sudo-calls").exists())

    def test_an_absent_directory_is_left_absent(self) -> None:
        result = self.clear("/cache/a/none")
        self.assertEqual((result.returncode, result.stdout), (0, "absent\n"))
        self.assertFalse((self.root / "mount" / "a" / "none").exists())
        self.assert_untouched()

    def test_a_link_out_of_the_mount_is_refused(self) -> None:
        (self.root / "mount" / "a" / "link").symlink_to(self.outside.parent)
        result = self.clear("/cache/a/link")
        self.assertEqual(result.returncode, 4)
        self.assertIn("not a cache directory of", result.stderr)
        self.assert_untouched()

    def test_a_link_to_a_directory_too_near_the_mount_is_refused(self) -> None:
        (self.root / "mount" / "a" / "up").symlink_to(self.root / "mount" / "a")
        result = self.clear("/cache/a/up")
        self.assertEqual(result.returncode, 4)
        self.assert_untouched()
        self.assertEqual(len(list(self.cache.iterdir())), 4)

    def test_a_running_serving_container_stops_the_clear(self) -> None:
        self.write_command("docker", "#!/bin/sh\necho c=experiments/x/arm.json\n")
        result = self.clear("/cache/a/b")
        self.assertEqual(result.returncode, 3)
        self.assertIn("a serving container is running: c=experiments/x/arm.json", result.stderr)
        self.assertEqual(len(list(self.cache.iterdir())), 4)

    def test_an_unreadable_docker_stops_the_clear(self) -> None:
        self.write_command("docker", "#!/bin/sh\necho 'permission denied' >&2\nexit 1\n")
        result = self.clear("/cache/a/b")
        self.assertEqual(result.returncode, 3)
        self.assertEqual(len(list(self.cache.iterdir())), 4)

    @unittest.skipIf(os.geteuid() == 0, "root removes files from a read-only directory")
    def test_files_the_user_cannot_remove_go_through_sudo(self) -> None:
        locked = self.cache / "v1" / "cuda-binaries"
        locked.chmod(0o555)
        self.addCleanup(lambda: locked.exists() and locked.chmod(0o755))
        refused = self.clear("/cache/a/b")
        self.assertEqual(refused.returncode, 5)
        self.assertEqual((self.root / "sudo-calls").read_text(), "-n find . -mindepth 1 -delete\n")
        self.assertTrue((locked / "one.bin").exists())
        self.write_command("sudo", "#!/bin/sh\nshift\nchmod -R u+w .\nexec \"$@\"\n")
        result = self.clear("/cache/a/b")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assert_untouched()

    def test_the_survey_reports_each_directory_and_tilelangs_entries(self) -> None:
        profile = self.profile(TILELANG_CACHE_DIR="/cache/a/b", TRITON_CACHE_DIR="/cache/a/none",
                               CUDA_CACHE_PATH="/cache/a/keep")
        directories, _, _ = spark.profile_caches(profile, spark.JIT_CACHES)
        before = sorted(path.as_posix() for path in self.root.rglob("*"))
        result = self.shell(spark.cache_survey_script(profile, directories, containers=True))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(path.as_posix() for path in self.root.rglob("*")), before)
        found = spark.parse_cache_survey(result.stdout, len(directories))
        self.assertEqual(found["running"], [])
        tilelang, nv_compute, triton = found["caches"]
        self.assertEqual(sorted(tilelang["entries"]), [("v1/cuda-binaries", 1), ("v1/linux-aarch64/kernels", 2)])
        self.assertEqual((tilelang["present"], nv_compute["present"], triton["present"]), (True, True, False))
        if self.gnu_find:
            self.assertEqual((tilelang["files"], tilelang["bytes"], nv_compute["files"]), (6, 60, 1))
            self.assertAlmostEqual(tilelang["newest"], self.sibling.stat().st_mtime, delta=120)


class StatusTest(unittest.TestCase):
    def test_status_makes_one_read_only_call_a_node_and_prints_each_cache(self) -> None:
        run = Run(spark.command_cache_status, surveys={"dgx1": lambda script: survey(script, entries=True)})
        self.assertEqual(run.code, 0)
        self.assertEqual([node for node, _ in run.scripts("survey")], ["dgx1", "dgx2", "dgx3"])
        self.assertEqual(run.scripts("clear"), [])
        self.assertNotIn("docker", run.calls[0][2])
        self.assertIn("profile: config/cluster.json on dgx1, dgx2, dgx3", run.out)
        self.assertIn("tilelang: TILELANG_CACHE_DIR=/cache/spark-ds41f/jit/tilelang\n", run.out)
        self.assertIn("vllm: VLLM_CACHE_ROOT, VLLM_CACHE_DIR=/cache/spark-ds41f/jit/vllm\n", run.out)
        self.assertIn("xdg: XDG_CACHE_HOME=/cache/spark-ds41f/jit (shown only; cache clear does not empty it)", run.out)
        text = "3.0 GiB in 3 files, modified 2026-11-07T07:20:00Z to 2026-11-08T11:06:40Z"
        for name in ("tilelang", "nv-compute", "triton", "vllm"):
            self.assertEqual(run.out.count(f"  {name}: {HOST_JIT}/{name}: {text}\n"), 3)
        self.assertEqual(run.out.count(f"  xdg: {HOST_JIT}: {text}\n"), 3)
        self.assertEqual(run.out.count("    v1/linux-aarch64/kernels: 412 entries\n"), 1)
        self.assertEqual(run.out.count("    v1/cuda-binaries: 7 entries\n"), 1)

    def test_absent_and_empty_directories_are_said_so(self) -> None:
        def answer(script: str) -> subprocess.CompletedProcess:
            return completed(stdout="cache\t0\tabsent\ncache\t1\tpresent\t0\t0\t0\t0\n")

        run = Run(spark.command_cache_status, surveys=dict.fromkeys(("dgx1", "dgx2", "dgx3"), answer),
                  profile=cluster({"TILELANG_CACHE_DIR": "/cache/a/b", "TRITON_CACHE_DIR": "/cache/a/c"}))
        self.assertEqual(run.code, 0)
        self.assertEqual(run.out.count("  tilelang: /h/r/cache/a/b: absent\n"), 3)
        self.assertEqual(run.out.count("  triton: /h/r/cache/a/c: present, no files\n"), 3)
        self.assertIn("nv-compute: the profile sets no CUDA_CACHE_PATH; skipped", run.out)

    def test_an_unreachable_node_fails_the_status_without_hiding_the_others(self) -> None:
        run = Run(spark.command_cache_status,
                  surveys={"dgx2": lambda _script: completed(255, stderr="ssh: connect to host dgx2: timed out")})
        self.assertEqual(run.code, 1)
        self.assertIn("ERROR: dgx2: unreachable: ssh: connect to host dgx2: timed out", run.err)
        self.assertIn("dgx1:\n", run.out)
        self.assertIn("dgx3:\n", run.out)
        self.assertNotIn("dgx2:\n", run.out)

    def test_a_node_whose_answer_cannot_be_read_fails_the_status(self) -> None:
        run = Run(spark.command_cache_status, surveys={"dgx3": lambda _script: completed(stdout="cache\t0\tabsent\n")})
        self.assertEqual(run.code, 1)
        self.assertIn("ERROR: dgx3: cannot inspect the caches: ", run.err)

    def test_status_keeps_the_default_profile(self) -> None:
        self.assertEqual(Run(spark.command_cache_status, cluster_config=None).code, 0)


class ClearTest(unittest.TestCase):
    def test_the_plan_names_every_directory_and_deletes_nothing(self) -> None:
        run = Run(spark.command_cache_clear, apply=False)
        self.assertEqual(run.code, 0)
        self.assertEqual(run.scripts("clear"), [])
        self.assertEqual([node for node, _ in run.scripts("survey")], ["dgx1", "dgx2", "dgx3"])
        for node in ("dgx1", "dgx2", "dgx3"):
            for name in ("tilelang", "nv-compute", "triton", "vllm"):
                self.assertIn(f"{node}: empty {HOST_JIT}/{name} ({name}: 3.0 GiB in 3 files, ", run.out)
        self.assertNotIn(f"empty {HOST_JIT} ", run.out)
        self.assertNotIn("would refuse", run.out)
        self.assertTrue(run.out.endswith("plan only; pass --apply to clear the caches\n"))

    def test_the_plan_keeps_the_default_profile_and_apply_needs_one_named(self) -> None:
        self.assertEqual(Run(spark.command_cache_clear, apply=False, cluster_config=None).code, 0)
        with mock.patch.object(spark, "remote_shell", side_effect=AssertionError("a remote call")), \
                self.assertRaises(SystemExit) as raised:
            spark.command_cache_clear(argparse.Namespace(apply=True, cluster_config=None, only=None, shape=None))
        self.assertIn("--cluster-config", str(raised.exception))

    def test_a_running_container_on_one_node_refuses_the_clear_on_every_node(self) -> None:
        running = {"dgx2": lambda script: survey(script, running="c=experiments/x/arm.json ")}
        run = Run(spark.command_cache_clear, apply=True, surveys=running)
        self.assertEqual(run.code, 1)
        self.assertEqual(run.scripts("clear"), [])
        self.assertIn("ERROR: c is running on dgx2, and a running engine has frozen its kernel set", run.err)
        self.assertIn("cluster stop comes first: bin/spark --cluster-config experiments/x/arm.json cluster stop --apply",
                      run.err)
        self.assertIn("nothing was cleared on any node", run.err)
        plan = Run(spark.command_cache_clear, apply=False, surveys=running)
        self.assertEqual((plan.code, plan.scripts("clear")), (0, []))
        self.assertIn("--apply would refuse: c is running on dgx2", plan.out)

    def test_a_container_running_on_several_nodes_is_named_once(self) -> None:
        def running(script: str) -> subprocess.CompletedProcess:
            return survey(script, running="c= ")

        run = Run(spark.command_cache_clear, apply=True, surveys={"dgx1": running, "dgx3": running})
        self.assertEqual((run.code, run.scripts("clear")), (1, []))
        self.assertIn("ERROR: c is running on dgx1, dgx3, and", run.err)
        self.assertIn("bin/spark --cluster-config PATH cluster stop --apply", run.err)

    def test_an_unreachable_node_refuses_the_clear_on_every_node(self) -> None:
        unreachable = {"dgx3": lambda _script: completed(255, stderr="ssh: connect to host dgx3: timed out")}
        run = Run(spark.command_cache_clear, apply=True, surveys=unreachable)
        self.assertEqual(run.code, 1)
        self.assertEqual(run.scripts("clear"), [])
        self.assertIn("ERROR: dgx3: unreachable: ssh: connect to host dgx3: timed out; "
                      "a clear covers every configured node or none", run.err)
        plan = Run(spark.command_cache_clear, apply=False, surveys=unreachable)
        self.assertEqual(plan.code, 0)
        self.assertIn("--apply would refuse: dgx3: unreachable", plan.out)

    def test_a_node_whose_containers_cannot_be_listed_refuses_the_clear(self) -> None:
        for answer in (lambda script: survey(script, docker=1, running="permission denied on docker.sock"),
                       lambda script: completed(stdout=survey(script).stdout.split("\n", 1)[1])):
            run = Run(spark.command_cache_clear, apply=True, surveys={"dgx1": answer})
            self.assertEqual((run.code, run.scripts("clear")), (1, []))
            self.assertIn("ERROR: dgx1: cannot tell whether an engine is running: ", run.err)

    def test_a_clear_issues_one_delete_script_for_each_cache_on_each_node(self) -> None:
        run = Run(spark.command_cache_clear, apply=True)
        self.assertEqual((run.code, run.err), (0, ""))
        self.assertEqual(len(run.scripts("survey")), 3)
        self.assertEqual([(node, script.splitlines()[1]) for node, script in run.scripts("clear")],
                         sorted((node, f"dir={HOST_JIT}/{name}") for node in ("dgx1", "dgx2", "dgx3")
                                for name in ("tilelang", "nv-compute", "triton", "vllm")))
        for node, kind, _ in run.calls[:3]:
            self.assertEqual(kind, "survey", f"{node} was changed before every node was read")
        self.assertEqual(run.out.count("removed 3 files, 2.0 KiB\n"), 12)
        self.assertIn(f"dgx2: {HOST_JIT}/triton (triton): removed 3 files, 2.0 KiB\n", run.out)
        self.assertNotIn("plan only", run.out)

    def test_only_selects_the_caches(self) -> None:
        run = Run(spark.command_cache_clear, apply=True, only=["triton", "tilelang", "triton"])
        self.assertEqual(run.code, 0)
        self.assertEqual([(node, script.splitlines()[1]) for node, script in run.scripts("clear")],
                         sorted((node, f"dir={HOST_JIT}/{name}") for node in ("dgx1", "dgx2", "dgx3")
                                for name in ("tilelang", "triton")))
        self.assertNotIn("nv-compute", run.out)
        self.assertNotIn("vllm", run.out)

    def test_only_an_unset_cache_clears_nothing_and_says_so(self) -> None:
        run = Run(spark.command_cache_clear, apply=True, only=["triton"],
                  profile=cluster({"TILELANG_CACHE_DIR": "/cache/a/b"}))
        self.assertEqual((run.code, run.calls), (0, []))
        self.assertIn("triton: the profile sets no TRITON_CACHE_DIR; skipped", run.out)
        self.assertIn("nothing to clear", run.out)

    def test_a_failed_delete_fails_the_command_and_the_rest_is_still_cleared(self) -> None:
        def failing(script: str) -> subprocess.CompletedProcess:
            if "/triton\n" in script:
                return completed(5, stderr="sudo: a password is required")
            return removed(script)

        run = Run(spark.command_cache_clear, apply=True, clears={"dgx2": failing})
        self.assertEqual(run.code, 1)
        self.assertEqual(len(run.scripts("clear")), 12)
        self.assertIn(f"ERROR: dgx2: {HOST_JIT}/triton (triton): not cleared: sudo: a password is required", run.err)
        self.assertEqual(run.out.count("removed 3 files"), 11)

    def test_an_absent_directory_is_not_an_error(self) -> None:
        run = Run(spark.command_cache_clear, apply=True, only=["vllm"],
                  clears={"dgx1": lambda _script: completed(stdout="absent\n")})
        self.assertEqual(run.code, 0)
        self.assertIn(f"dgx1: {HOST_JIT}/vllm (vllm): absent, nothing to remove\n", run.out)

    def test_a_delete_that_reports_nothing_is_an_error(self) -> None:
        run = Run(spark.command_cache_clear, apply=True, only=["vllm"], clears={"dgx1": lambda _script: completed()})
        self.assertEqual(run.code, 1)
        self.assertIn(f"ERROR: dgx1: {HOST_JIT}/vllm (vllm): not cleared: exit 0", run.err)


class ParserTest(unittest.TestCase):
    def test_the_commands_are_wired(self) -> None:
        status = spark.parser().parse_args(["cache", "status"])
        self.assertEqual((status.func.__name__, status.cluster_config), ("command_cache_status", None))
        clear = spark.parser().parse_args(["--cluster-config", "x.json", "cache", "clear"])
        self.assertEqual((clear.func.__name__, clear.apply, clear.only), ("command_cache_clear", False, None))
        named = spark.parser().parse_args(["cache", "clear", "--only", "tilelang", "nv-compute", "--only", "vllm",
                                           "--apply"])
        self.assertEqual((named.apply, named.only), (True, ["tilelang", "nv-compute", "vllm"]))

    def test_only_takes_cache_names(self) -> None:
        for name in ("xdg", "huggingface", "/cache"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                spark.parser().parse_args(["cache", "clear", "--only", name])


if __name__ == "__main__":
    unittest.main()
