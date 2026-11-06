#!/usr/bin/env python3
"""One bounded remote source proof; never a C43 native gate or serving rerun."""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import traceback

PIN = "2c4b23afcd834ba2adf6a15ca434e59fa4f49f74"
BASE = "19f2c20ed4d64ae22bd009ceacadd85e2a81e354"
HEAD = "99029e6cbdb966ff198dd13902a0825e8d507e27"
TREE = "082bda0f13e94290f462eaf55d143cdb7931f9e5"
PREFIX_HASH = "05586bfb7487c6436f75bc240e934ff7d1c2b81877f8c0014ae985a9f9d6245a"
DRAFT_HASH = "e7950e488a1b874427a65e3e57e9efdf252a593e5679d93a4d6a1d3433020b92"
NEW_HASH = "7c15eb3a5704e43b08ca06f91f7a4145d59efdbf3e7084f0dce025b09b6a91c0"
SERIES = "experiments/2026-10-26-spark-ds41f-candidate/candidate-43/vllm/series"
LOCK = "experiments/2026-10-26-spark-ds41f-candidate/candidate-43/upstreams.lock.json"
PRODUCER = "tests/v1/worker/test_gpu_tilelang_logprob_softmax.py"
FILES = [PRODUCER, "tests/v1/worker/test_gpu_verified_logprob_token_ids.py",
         "vllm/v1/worker/gpu/sample/logprob.py", "vllm/v1/engine/async_llm.py",
         "vllm/v1/engine/core_client.py", "tests/v1/engine/test_request_submission_cleanup.py",
         "tests/v1/engine/test_utility_call_cleanup.py"]
GIT = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-c", "pack.threads=1",
       "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
IDENTITY = ["-c", "user.name=Christopher Owen", "-c",
            "user.email=3221756+christopherowen@users.noreply.github.com"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    args = p.parse_args()
    if sys.platform != "linux" or os.uname().nodename != "spark-0016":
        raise SystemExit("explicit sp8 CPU-only scope required; no workstation fallback")
    work = args.work.resolve()
    if (work / "receipt.json").exists():
        raise SystemExit("prior source receipt exists; preserve it")
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    os.nice(10)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        os.environ[name] = "1"
    os.environ.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", PYTHONDONTWRITEBYTECODE="1")
    r = {"started": now(), "purpose": "source-only fixture correction proof", "steps": [],
         "prefix_deployment_commit": PIN, "native_gate_GPU_or_serving_work": False}
    claim = Path.home() / "spark-lab/reservations/copy-default-screening-c46-source.json"
    claim_data = None
    locks = []

    def save():
        (work / "receipt.json").write_text(json.dumps(r, indent=2) + "\n")

    def run(argv, cwd=None, timeout=120, required=True, env=None, input_data=None):
        index = len(r["steps"])
        step = {"argv": list(map(str, argv)), "cwd": str(cwd) if cwd else None, "started": now()}
        r["steps"].append(step)
        save()
        try:
            q = subprocess.run(step["argv"], cwd=cwd, capture_output=True, timeout=timeout,
                               env=env, input=input_data)
        except BaseException as error:
            step.update(error=type(error).__name__ + ": " + str(error), ended=now())
            save()
            raise
        for name, data in (("stdout", q.stdout), ("stderr", q.stderr)):
            out = work / f"step-{index:04d}-{name}.log"
            out.write_bytes(data)
            step[name] = {"path": out.name, "bytes": len(data), "sha256": sha(data)}
        step.update(exit=q.returncode, ended=now())
        save()
        if required and q.returncode:
            raise RuntimeError(f"step {index} exited {q.returncode}; retain its logs")
        return q

    def git(repo, *argv, **kw):
        return run(GIT + ["-C", str(repo), *argv], **kw)

    def text(repo, *argv):
        return git(repo, *argv).stdout.decode().strip()

    def interruption(signum, frame):
        raise RuntimeError(f"source supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interruption)
    signal.signal(signal.SIGALRM, interruption)
    signal.alarm(840)
    code = 1
    try:
        cgline = next(s for s in Path("/proc/self/cgroup").read_text().splitlines() if s.startswith("0::"))
        cg = Path("/sys/fs/cgroup") / cgline.split("::", 1)[1].lstrip("/")
        actual = {n: (cg / n).read_text().strip() for n in ("memory.max", "memory.swap.max", "cpu.max", "pids.max")}
        quota, period = map(int, actual["cpu.max"].split())
        if (actual["memory.max"] != str(8 * 1024**3) or actual["memory.swap.max"] != "0"
                or quota > period or int(actual["pids.max"]) > 64):
            raise RuntimeError("source cgroup limits are not actually enforced")
        r["resources"] = {"cgroup": str(cg), "effective": actual,
                          "affinity": sorted(os.sched_getaffinity(0)), "address_space_bytes": 8 * 1024**3}
        v = os.statvfs(work)
        r["free_disk_bytes"] = v.f_bavail * v.f_frsize
        mem = {s.split(":")[0]: int(s.split()[1]) for s in Path("/proc/meminfo").read_text().splitlines()
               if s.split(":")[0] in ("MemAvailable", "SwapFree")}
        r["memory_kib"] = mem
        if r["free_disk_bytes"] < 12 * 1024**3 or mem["MemAvailable"] < 16 * 1024**2:
            raise RuntimeError("insufficient remote disk or memory; no local fallback")
        for path in (Path.home() / "spark-cpu-work.lock", Path.home() / "spark-lab/offline-cpu.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            f = path.open("a")
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(f)
        held = {}
        for file in (Path.home() / "spark-lab/reservations").glob("*.json"):
            value = json.loads(file.read_text())
            held[str(file)] = value
            if value.get("kind") != "cpu-only":
                raise RuntimeError("foreign non-CPU host reservation; defer")
        for name in ("spark-hold.json", "spark-request.json", "spark3-request.json"):
            file = Path.home() / name
            if file.exists():
                raise RuntimeError("host hold/request exists; preserve it and defer")
        r["host_reservations_before"] = held
        gpu = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], timeout=10)
        if gpu.stdout.strip():
            raise RuntimeError("GPU work active; no concurrent source preparation")
        ids = run(["docker", "ps", "-q"], timeout=10).stdout.decode().split()
        containers = []
        for cid in ids:
            x = json.loads(run(["docker", "inspect", cid], timeout=10).stdout)[0]
            h = x["HostConfig"]
            if (h.get("DeviceRequests") or not h.get("CpuQuota") or h["CpuQuota"] > 2 * h.get("CpuPeriod", 0)
                    or not h.get("Memory") or h["Memory"] > 32 * 1024**3 or h.get("MemorySwap") != h["Memory"]):
                raise RuntimeError("existing container is not a bounded CPU-only job")
            containers.append({"id": x["Id"], "name": x["Name"],
                               "cpu_quota": h["CpuQuota"], "cpu_period": h["CpuPeriod"],
                               "memory_bytes": h["Memory"], "memory_swap_bytes": h["MemorySwap"]})
        r["concurrent_CPU_containers"] = containers
        claim_data = {"holder": "copy-default-screening-c46-source", "kind": "cpu-only", "cpus": 1,
                      "memory_gib": 8, "since": now(), "pid": os.getpid(), "workdir": str(work),
                      "rule": "Source-only canonical replay; both CPU locks held; no GPU/serving/profile changes."}
        with claim.open("x") as f:
            json.dump(claim_data, f, indent=2)
            f.write("\n")
        r["own_claim"] = claim_data
        draft = args.draft.read_bytes()
        if sha(draft) != DRAFT_HASH:
            raise RuntimeError("published draft bytes differ")
        packet_git = work / "deployment.git"
        existing = Path.home() / "projects/spark-ds41f"
        run(GIT + ["clone", "--shared", "--bare", "--no-hardlinks", str(existing), str(packet_git)])
        git(packet_git, "remote", "remove", "origin")
        git(packet_git, "remote", "add", "deployment", "https://github.com/christopherowen/spark-ds41f.git")
        if git(packet_git, "cat-file", "-e", PIN + "^{commit}", required=False).returncode:
            git(packet_git, "fetch", "--no-tags", "--depth=1", "--filter=blob:none", "deployment", PIN, timeout=180)

        def blob(name):
            oid = text(packet_git, "rev-parse", PIN + ":" + name)
            if git(packet_git, "cat-file", "-e", oid, required=False).returncode:
                git(packet_git, "fetch", "--no-tags", "--filter=blob:none", "deployment", oid, timeout=90)
            return git(packet_git, "cat-file", "blob", oid).stdout

        lock = json.loads(blob(LOCK))
        config = lock["sources"]["vllm"]
        series = blob(SERIES)
        paths = []
        d = hashlib.sha256()
        for c in config.get("applied_upstream_commits", []):
            d.update(c["revision"].encode() + b"\0")
        d.update(series)
        packet = work / "packet-files"
        material = []
        for name in series.decode().splitlines():
            if not name.strip() or name.lstrip().startswith("#"):
                continue
            destination = (packet / SERIES).parent.joinpath(name).resolve()
            if not destination.is_relative_to(packet):
                raise RuntimeError("patch path escapes minimal packet")
            rel = destination.relative_to(packet).as_posix()
            data = blob(rel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            paths.append(destination)
            material.append({"name": name, "path": rel, "bytes": len(data), "sha256": sha(data)})
            d.update(name.encode() + b"\0" + data)
        if len(paths) != 576 or d.hexdigest() != PREFIX_HASH:
            raise RuntimeError("published prefix count/hash differs")
        r["prefix"] = {"files": material, "series_sha256": sha(series), "patchset": d.hexdigest()}
        candidates = sorted((existing / ".work/build").glob("*/src/vllm"))
        cache = None
        for candidate in candidates:
            if git(candidate, "cat-file", "-e", BASE + "^{commit}", required=False, timeout=10).returncode == 0:
                listing = git(candidate, "ls-tree", "-r", BASE, required=False, timeout=20)
                if listing.returncode:
                    continue
                oids = b"\n".join(line.split(b"\t", 1)[0].split()[2]
                                  for line in listing.stdout.splitlines()
                                  if line.split(b"\t", 1)[0].split()[1] == b"blob") + b"\n"
                available = git(candidate, "cat-file", "--batch-check=%(objectname) %(objecttype)",
                                input_data=oids, required=False, timeout=20)
                if available.returncode == 0 and b" missing" not in available.stdout:
                    cache = candidate
                    break
        if cache is None:
            raise RuntimeError("pinned base absent from read-only remote caches; no full clone fallback")
        source = work / "source"
        run(GIT + ["clone", "--shared", "--no-checkout", "--no-hardlinks", str(cache), str(source)])
        git(source, "remote", "remove", "origin")
        git(source, "remote", "add", "upstream", config["upstream"])
        git(source, "remote", "add", "origin", config["contribution_remote"])
        git(source, "checkout", "-b", "feat/worker-test-isolation-rework", BASE)
        run(GIT + IDENTITY + ["-C", str(source), "am", "--quiet", "--committer-date-is-author-date", *map(str, paths)], timeout=300)
        if text(source, "rev-parse", "HEAD") != HEAD or text(source, "rev-parse", "HEAD^{tree}") != TREE:
            raise RuntimeError("canonical prefix identity differs")
        if text(source, "status", "--porcelain"):
            raise RuntimeError("canonical source is dirty")
        for name in FILES:
            p = work / "before-source" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((source / name).read_bytes())
        git(source, "apply", "--check", str(args.draft.resolve()))
        git(source, "apply", str(args.draft.resolve()))
        if text(source, "diff", "--name-only") != PRODUCER or sha((source / PRODUCER).read_bytes()) != NEW_HASH:
            raise RuntimeError("draft is not the exact reviewed one-file correction")
        ast.parse((source / PRODUCER).read_text())
        for name in FILES[1:]:
            if (source / name).read_bytes() != (work / "before-source" / name).read_bytes():
                raise RuntimeError("runtime/consumer/frontend bytes changed")
        git(source, "diff", "--check")
        date = now()
        message = ("Isolate lazy-reference poisoning in the worker test\n\n"
                   f"Upstream base: {BASE}\nSource base: C43 {HEAD}\n"
                   "Rationale: restore the absent dictionary entry without resolving a lazy reference.\n"
                   "Quality: test fixture isolation only; all numerical/routing assertions, runtime bytes,\n"
                   "consumer checks, manifest and grouping remain unchanged.\n"
                   "Tests: fresh 576-patch canonical prefix identity; exact one-file reviewed diff;\n"
                   "Python syntax and whitespace. Actual-image pytest witness and native gates pending.\n"
                   "Upstream status: local, not submitted.\n")
        git(source, "add", PRODUCER)
        env = os.environ.copy()
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        run(GIT + IDENTITY + ["-C", str(source), "commit", "-m", message], env=env)
        new_head, new_tree = text(source, "rev-parse", "HEAD"), text(source, "rev-parse", "HEAD^{tree}")
        patch = git(source, "format-patch", "--stdout", "--full-index", "-1", "HEAD").stdout
        patch_path = work / "0001-isolate-lazy-test-reference.patch"
        patch_path.write_bytes(patch)
        for name in FILES:
            p = work / "after-source" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((source / name).read_bytes())
        verification = work / "reapply"
        run(GIT + ["clone", "--shared", "--no-checkout", "--no-hardlinks", str(source), str(verification)])
        git(verification, "checkout", "-b", "feat/worker-test-isolation-reapply", HEAD)
        run(GIT + IDENTITY + ["-C", str(verification), "am", "--quiet", "--committer-date-is-author-date", str(patch_path)])
        if text(verification, "rev-parse", "HEAD") != new_head or text(verification, "rev-parse", "HEAD^{tree}") != new_tree:
            raise RuntimeError("formatted correction does not reproduce exact identity")
        r["result"] = {"head": new_head, "tree": new_tree, "patch_sha256": sha(patch), "patch_path": str(patch_path),
                       "producer_sha256": NEW_HASH, "changed_files": [PRODUCER], "author_committer_date": date,
                       "actual_image_pytest_witness": "pending", "native_qualification": "not started"}
        r["passed"] = True
        code = 0
    except BaseException as error:
        r["error"] = type(error).__name__ + ": " + str(error)
        (work / "failure-traceback.log").write_text(traceback.format_exc())
        r["passed"] = False
    finally:
        signal.alarm(0)
        if claim_data is not None and claim.exists() and json.loads(claim.read_text()) == claim_data:
            claim.unlink()
        r["ended"] = now()
        save()
        print(json.dumps({k: r.get(k) for k in ("passed", "error", "result", "ended")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
