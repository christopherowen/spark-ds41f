"""Remote CPU-only canonical source derivation; no builds, imports or hardware work."""
from pathlib import Path
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback

BASE = "19f2c20ed4d64ae22bd009ceacadd85e2a81e354"
HEAD = "5f926df6d3126266a9c88c82c06544060008c33c"
TREE = "ca1496c6481dc5bc4b4234b743e4ca023a6c7597"
PATCHSET = "c7cdd12575661dc07642c9b4207a224f2b39bd913a259336dd28ee2af945390b"
DELTA_SHA = "23ff853ce80250edb63eeea9ab5f0c3f9d104ca8ac933841076ef82bf86835a7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    work, packet, home = args.work.resolve(), args.packet.resolve(), Path.home()
    if not work.is_dir() or (work / "receipt.json").exists() or (work / "source").exists():
        raise SystemExit("use one fresh source-output directory; preserve every prior attempt")
    claim = home / "spark-lab/reservations/copy-default-metadata-source.json"
    owned, locks = None, []
    result = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "packet_commit": args.commit, "steps": [], "passed": False,
              "purpose": "performance enhancement; source derivation only",
              "GPU_build_serving_or_native_measurement": False}
    def save():
        (work / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    def run(argv, name, cwd=None):
        number = len(result["steps"])
        process = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=120)
        prefix = f"{number:04d}-{name}"
        (work / (prefix + ".stdout")).write_text(process.stdout)
        (work / (prefix + ".stderr")).write_text(process.stderr)
        result["steps"].append({"name": name, "argv": argv, "exit": process.returncode,
                                "stdout": prefix + ".stdout", "stderr": prefix + ".stderr"})
        if process.returncode:
            save(); raise RuntimeError(name + " exited " + str(process.returncode))
        return process.stdout.strip()
    try:
        assert sys.platform == "linux" and os.uname().nodename == "spark-0016"
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3,) * 2)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MAX_JOBS"):
            os.environ[name] = "1"
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", GIT_TERMINAL_PROMPT="0")
        cgroup = Path("/sys/fs/cgroup") / next(x.split("::", 1)[1].lstrip("/") for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
        limits = {key: (cgroup / key).read_text().strip() for key in ("cpu.max", "memory.max", "memory.swap.max")}
        assert limits == {"cpu.max": "100000 100000", "memory.max": str(8 * 1024**3), "memory.swap.max": "0"}
        result["effective_limits"] = limits
        assert shutil.disk_usage(work).free > 20 * 1024**3
        for name in ("spark-hold.json", "spark-request.json", "spark3-request.json", "spark-node-hold.json", "spark-cpu-hold.json"):
            assert not (home / name).exists(), name
        for path in (home / "spark-cpu-work.lock", home / "spark-lab/offline-cpu.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a")
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(stream)
        for path in (home / "spark-lab/reservations").glob("*.json"):
            assert json.loads(path.read_text()).get("kind") == "cpu-only", str(path)
        source_cache = home / "spark-work/worker-test-isolation/c46-source-attempt-1/source"
        assert (source_cache / ".git").exists(), str(source_cache)
        owned = {"holder": "copy-default-metadata-source", "kind": "cpu-only", "cpus": 1,
                 "memory_gib": 8, "pid": os.getpid(), "since": result["started"],
                 "active_git_alternates_target": str(source_cache / ".git/objects"),
                 "rule": "Fresh C40 plus exact metadata patch source replay only; no GPU/build/serving/old measurement"}
        claim.parent.mkdir(parents=True, exist_ok=True)
        with claim.open("x") as stream:
            json.dump(owned, stream)
        git_base = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-c", "pack.threads=1"]
        def git(directory, *argv, name):
            return run(git_base + ["-C", str(directory), *argv], name)
        assert not git(packet, "status", "--porcelain", name="private-input-clean-before-update")
        git(packet, "fetch", "--no-tags", "--depth=1", "--filter=blob:none", "deployment", args.commit,
            name="fetch-exact-published-source-plan")
        git(packet, "sparse-checkout", "add", "--no-cone",
            "/experiments/2026-11-05-pr-screening/sources/pr-919/",
            "/experiments/2026-11-06-drafter-metadata-transfer/", name="source-plan-sparse-inputs")
        git(packet, "checkout", "-b", "feat/metadata-transfer-input-1", args.commit,
            name="published-private-input")
        assert git(packet, "rev-parse", "HEAD", name="packet-identity") == args.commit
        assert not git(packet, "status", "--porcelain", name="packet-clean")
        assert hashlib.sha256(Path(__file__).read_bytes()).digest() == hashlib.sha256(
            (packet / "experiments/2026-11-06-drafter-metadata-transfer/source_prepare.py").read_bytes()).digest()
        sys.path.insert(0, str(packet))
        from spark_ds41f import build
        series = packet / "experiments/2026-10-26-spark-ds41f-candidate/candidate-40/vllm/series"
        lock = json.loads((series.parent.parent / "upstreams.lock.json").read_text())["sources"]["vllm"]
        assert lock["revision"] == BASE and not lock.get("applied_upstream_commits")
        assert build.patchset_sha256(lock) == PATCHSET
        patch_names = build.active_series_patches(series)
        patches = [(series.parent / name).resolve() for name in patch_names]
        assert all(path.is_relative_to(packet) and path.is_file() for path in patches)
        delta = packet / "experiments/2026-11-05-pr-screening/sources/pr-919/0688-replay-drafter-metadata.patch"
        assert hashlib.sha256(delta.read_bytes()).hexdigest() == DELTA_SHA
        source = work / "source"
        run(git_base + ["clone", "--shared", "--no-checkout", "--quiet", str(source_cache), str(source)], "fresh-clone")
        git(source, "switch", "--quiet", "--create", "feat/drafter-metadata-transfer", BASE, name="pinned-base")
        for index, patch in enumerate(patches, 1):
            git(source, "-c", "user.name=" + build.PATCH_COMMITTER[0], "-c", "user.email=" + build.PATCH_COMMITTER[1],
                "am", "--quiet", "--committer-date-is-author-date", str(patch), name="base-patch")
            (work / "progress.json").write_text(json.dumps({"phase": "C40 canonical replay", "patch": index,
                "total": len(patches), "updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
        assert git(source, "rev-parse", "HEAD", name="C40-head") == HEAD
        assert git(source, "rev-parse", "HEAD^{tree}", name="C40-tree") == TREE
        result["verified_C40"] = {"head": HEAD, "tree": TREE, "patchset": PATCHSET, "patches": len(patches)}
        save()
        git(source, "-c", "user.name=" + build.PATCH_COMMITTER[0], "-c", "user.email=" + build.PATCH_COMMITTER[1],
            "am", "--quiet", "--committer-date-is-author-date", str(delta), name="authored-metadata-delta")
        changed = git(source, "diff", "--name-only", HEAD, "HEAD", name="changed-files").splitlines()
        plan = json.loads((delta.parent / "c40-transfer-plan.json").read_text())
        expected = plan["authored_delta"]["runtime_files"] + plan["authored_delta"]["test_files"]
        assert sorted(changed) == sorted(expected), changed
        for name in changed:
            ast.parse((source / name).read_text(), filename=name)
        git(source, "diff", "--check", HEAD, "HEAD", name="delta-whitespace")
        assert not git(source, "status", "--porcelain", name="derived-source-clean")
        destination = packet / "experiments/2026-11-06-drafter-metadata-transfer/vllm"
        names = [os.path.relpath(path, destination) for path in patches] + [delta.name]
        series_bytes = ("\n".join(names) + "\n").encode()
        digest = hashlib.sha256(series_bytes)
        for name, path in zip(names, [*patches, delta]):
            digest.update(name.encode()); digest.update(b"\0"); digest.update(path.read_bytes())
        derived = work / "derived"; derived.mkdir()
        (derived / "series").write_bytes(series_bytes)
        shutil.copyfile(delta, derived / delta.name)
        result["derived"] = {"head": git(source, "rev-parse", "HEAD", name="new-head"),
            "tree": git(source, "rev-parse", "HEAD^{tree}", name="new-tree"),
            "series_sha256": hashlib.sha256(series_bytes).hexdigest(), "patchset_sha256": digest.hexdigest(),
            "patches": len(names), "changed_files": changed, "authored_patch_sha256": DELTA_SHA,
            "parse_and_delta_whitespace_passed": True,
            "canonical_source": str(source), "required_alternate_target": str(source_cache / ".git/objects")}
        result["native_build_and_host_graph_tests"] = "pending; not executed by this source derivation"
        result["passed"] = True
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        (work / "error.log").write_text(traceback.format_exc())
    finally:
        if owned is not None and claim.exists() and json.loads(claim.read_text()) == owned:
            claim.unlink(); result["matching_claim_removed"] = True
        result["ended"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print(json.dumps({key: result.get(key) for key in ("passed", "error", "derived", "ended")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
