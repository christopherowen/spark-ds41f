#!/usr/bin/env python3
"""C46's exact DGX lease and one bound restoration watchdog; no pool scheduling API."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab
import c46_policy as policy
from c46_management import ManagementLatch

SCRIPT = str(Path(__file__).resolve())
_WATCHDOGS = {}


class WaitForOwnedJob(RuntimeError):
    pass


def matches(hold, window, holder, token, review=False):
    return (isinstance(hold, dict) and hold.get("window") == window
            and hold.get("c46_owner_token") == token and hold.get("c46_holder") == holder
            and (review or hold.get("holder") == holder))


@contextmanager
def mutation_lock():
    with lab.HOLD.with_suffix(".c46.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def exclusive_create(path, payload):
    """Publish a complete JSON inode only if no hold exists; a competing hold is never replaced."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".c46-hold-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(payload, indent=2) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def update(window, holder, token, change, review=False):
    with mutation_lock():
        hold = lab.read_hold()
        if not matches(hold, window, holder, token, review):
            raise RuntimeError("exact C46 lease changed; foreign hold untouched")
        change(hold)
        lab.write_json_atomic(lab.HOLD, hold)
        return hold


def beat(window, holder, token, note=None):
    def change(hold):
        hold["heartbeat"] = lab.now()
        if note:
            hold["current"] = note
    return update(window, holder, token, change)


def hardware_review(window, holder, token, management):
    def change(hold):
        hold.update(holder="candidate-round-hardware-review",
                    rule="C46 management/guard telemetry loss: no restoration launch or new hardware work until isolated and explicitly authorized",
                    **management.snapshot())
    return update(window, holder, token, change, review=True)


def clean_sources(commit, management):
    records = {}
    for host in policy.NODES:
        directory = "/home/swank/projects/spark-ds41f-copy-default" if host == "dgx4" else "/home/swank/projects/spark-ds41f"
        code = "import json,subprocess;d=" + repr(directory) + ";print(json.dumps({'commit':subprocess.check_output(['git','-C',d,'rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','-C',d,'status','--porcelain','--untracked-files=all'],text=True)}))"
        records[host] = json.loads(management.ssh(host, ["python3", "-c", code], "clean-published-source"))
        if records[host]["commit"] != commit or records[host]["dirty"]:
            raise RuntimeError(host + ": source must be clean at the exact published execution commit")
    return records


def open_window(token, note, commit, management):
    if lab.read_hold() or lab.REQUEST.exists() or lab.LEGACY_REQUEST.exists():
        raise RuntimeError("existing hold/request; no C46 window created")
    # Reuse the existing publication guard, but its node metadata reads must
    # share C46's bounded, sticky transport handling.
    original_ssh = lab.spark.run_ssh
    def guard_read(nodes, node, *argv):
        text = management.ssh(lab.spark.ssh_target(nodes, node), [str(value) for value in argv], "publication-identity")
        return subprocess.CompletedProcess(argv, 0, text, "")
    try:
        lab.spark.run_ssh = guard_read
        problems = lab.published_problems()
    finally:
        lab.spark.run_ssh = original_ssh
    if problems:
        raise RuntimeError("not published: " + "; ".join(problems))
    preflight = ROOT / "results/private/lab" / ("c46-open-live-" + token + ".log")
    preflight.parent.mkdir(parents=True, exist_ok=True)
    code = _local([sys.executable, "bin/spark", "--cluster-config", policy.RESTORE_PROFILE, "doctor", "--live"],
                  preflight, management, "initial-live-guard")
    if code or "live cluster matches" not in preflight.read_text() or not lab.idle_for(30):
        raise RuntimeError("named restoration service is not live and idle")
    initial_restore = _ranks(management)
    if len(initial_restore) != 3 or any(item.get("running") is not True or item.get("image") != policy.RESTORE_IMAGE
                                      or item.get("profile") != policy.RESTORE_PROFILE for item in initial_restore):
        raise RuntimeError("live all-rank service is not the immutable named C36 restoration image")
    sources = clean_sources(commit, management)
    if lab.REQUEST.exists() or lab.LEGACY_REQUEST.exists() or management.latched:
        raise RuntimeError("request or management loss during preparation; no window created")
    hold = lab.new_hold(0, note)
    holder, window = hold["holder"], hold["window"]
    hold.update(c46_owner_token=token, c46_holder=holder, execution_commit=commit)
    exclusive_create(lab.HOLD, hold)
    argv = [sys.executable, SCRIPT, "watchdog", "--window", window, "--holder", holder,
            "--owner-token", token, "--commit", commit, "--restore-profile", policy.RESTORE_PROFILE]
    logfile = ROOT / "results/private/lab" / ("c46-watchdog-" + window + ".log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        with logfile.open("x") as output:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        _WATCHDOGS[child.pid] = child
        update(window, holder, token, lambda current: current.update(c46_watchdog={"pid": child.pid, "argv": argv}))
    except BaseException:
        # No workload has begun; retain the exact lease for conservative inspection.
        raise
    return {"hold": hold, "watchdog_pid": child.pid, "watchdog_argv": argv,
            "watchdog_script": SCRIPT, "sources": sources, "initial_restore_identity": initial_restore}


def _ranks(management):
    result = []
    name = lab.spark.read_cluster(policy.RESTORE_PROFILE)["container"]["name"]
    code = "import json,subprocess,sys;p=subprocess.run(['docker','inspect',sys.argv[1]],capture_output=True,text=True);x=json.loads(p.stdout)[0] if p.returncode==0 else None;print(json.dumps({'id':x['Id'],'image':x['Image'],'running':x['State']['Running'],'started':x['State']['StartedAt'],'restarts':x['RestartCount'],'profile':x['Config'].get('Labels',{}).get('spark.cluster-config')} if x else {'running':False,'inspect_exit':p.returncode,'inspect_error':p.stderr}))"
    for host in policy.NODES[:3]:
        item = json.loads(management.ssh(host, ["python3", "-c", code, name], "restore-immutable-identity"))
        if item.get("inspect_exit") and "no such" not in item.get("inspect_error", "").lower():
            raise RuntimeError(host + ": runtime metadata inspection failed; no absent engine inferred")
        if not item.get("inspect_exit") and (not re.fullmatch(r"[0-9a-f]{64}", item.get("id", ""))
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", item.get("image", ""))
                or type(item.get("running")) is not bool):
            raise RuntimeError(host + ": malformed runtime identity; no restoration state inferred")
        result.append({"host": host, **item})
    return result


def _local(argv, path, management, phase, timeout=180):
    with path.open("x") as output:
        try:
            result = subprocess.run(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            management.latch("management_metadata_timeout", phase, detail=error)
            raise
    management.observe_file(path, phase)
    return result.returncode


def verify_restored(out, management, label):
    before = _ranks(management)
    path = out / (label + "-doctor.log")
    argv = [sys.executable, "bin/spark", "--cluster-config", policy.RESTORE_PROFILE, "doctor", "--live"]
    code = _local(argv, path, management, "restore-live-doctor")
    after = _ranks(management)
    passed = (code == 0 and "live cluster matches" in path.read_text() and before == after
              and len(after) == 3 and len({item.get("id") for item in after}) == 3
              and all(item["running"] is True and item["image"] == policy.RESTORE_IMAGE
                  and item["profile"] == policy.RESTORE_PROFILE for item in after))
    result = {"passed": passed, "argv": argv, "doctor_exit": code, "before": before, "after": after}
    lab.write_json_atomic(out / (label + "-identities.json"), result)
    return passed


def _alive_restore(hold):
    process = hold.get("c46_restore_process", {})
    try:
        stat = Path("/proc", str(process["pid"]), "stat").read_text().rsplit(")", 1)[1].split()
        return stat[0] != "Z" and int(stat[19]) == process["start_ticks"]
    except (KeyError, FileNotFoundError, ProcessLookupError):
        return False


def _stop_watchdog(hold):
    watchdog = hold.get("resource_scope", {}).get("watchdog", hold.get("c46_watchdog", {}))
    child = _WATCHDOGS.get(watchdog.get("pid"))
    if child is not None and child.poll() is None:
        child.terminate(); child.wait(timeout=5)


def close_window(window, holder, token, commit, out, management, watchdog=False):
    # Serialize this parent and its own watchdog only. The exact hold is still
    # rechecked after waiting; this lock grants no authority over another holder.
    with lab.HOLD.with_suffix(".c46-restore.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _close_window(window, holder, token, commit, out, management, watchdog)
        except BaseException:
            if management.latched and matches(lab.read_hold(), window, holder, token, True):
                hardware_review(window, holder, token, management)
            raise


def _close_window(window, holder, token, commit, out, management, watchdog=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    hold = lab.read_hold()
    if not matches(hold, window, holder, token, review=True):
        raise RuntimeError("foreign/replaced hold; no C46 restoration action")
    if hold.get("execution_commit") != commit:
        raise RuntimeError("restoration commit differs from the exact C46 lease")
    if management.latched or hold.get("management_loss_latched") or hold.get("holder") != holder:
        if management.latched:
            hardware_review(window, holder, token, management)
        raise RuntimeError("sticky hardware review: no restoration launch")
    for host in policy.NODES:
        management.ssh(host, ["true"], "pre-restoration-management", probe=True, timeout=15)
    clean_sources(commit, management)
    # Never release capacity while the isolated DGX4 job is still active.
    active = management.ssh("dgx4", ["docker", "ps", "-q"], "builder-quiescence")
    gpu = management.ssh("dgx4", ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], "builder-quiescence")
    builder_active = bool(active.strip() or gpu.strip())
    if not verify_restored(out, management, "before-restore"):
        if management.latched:
            hardware_review(window, holder, token, management)
            raise RuntimeError("management/telemetry loss; no restoration launch")
        if _alive_restore(lab.read_hold() or {}):
            raise WaitForOwnedJob("an exact prior restoration process is still alive; no duplicate launch")
        if (lab.read_hold() or {}).get("c46_restore_process"):
            raise RuntimeError("a prior restoration attempt exists; preserve its failure rather than launch again")
        argv = [sys.executable, "bin/spark", "--cluster-config", policy.RESTORE_PROFILE, "cluster", "start", "--replace"]
        path = out / "restore-start-plan.log"
        code = _local(argv, path, management, "restoration-plan")
        plan = path.read_text()
        if code or "--apply would refuse" in plan or "shape:   CHANGES" in plan or "live:" not in plan or "shape:" not in plan:
            raise RuntimeError("named restoration plan refused; no shape bypass")
        if management.latched or not matches(lab.read_hold(), window, holder, token):
            raise RuntimeError("lease or management changed before restoration launch")
        with (out / "restore-start.log").open("x") as output:
            child = subprocess.Popen([*argv, "--apply"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            stat = Path("/proc", str(child.pid), "stat").read_text().rsplit(")", 1)[1].split()
            update(window, holder, token, lambda current: current.update(c46_restore_process={"pid": child.pid,
                "start_ticks": int(stat[19]), "argv": [*argv, "--apply"], "commit": commit}))
            code = child.wait()
        management.observe_file(out / "restore-start.log", "restoration-start")
        if code or management.latched or not verify_restored(out, management, "after-restore"):
            raise RuntimeError("all-rank immutable C36 restoration did not verify")
    if management.latched:
        hardware_review(window, holder, token, management)
        raise RuntimeError("management loss is sticky after recovery")
    if builder_active:
        raise WaitForOwnedJob("C36 is restored but DGX4 job is still active; retain exact lease until its safe end")
    with mutation_lock():
        final = lab.read_hold()
        if not matches(final, window, holder, token) or final.get("management_loss_latched"):
            raise RuntimeError("lease changed before release; foreign hold untouched")
        if not watchdog:
            _stop_watchdog(final)
        if not matches(lab.read_hold(), window, holder, token):
            raise RuntimeError("lease changed while stopping its watchdog; replacement retained")
        lab.HOLD.unlink()
    return True


def watchdog_main(args):
    if args.restore_profile != policy.RESTORE_PROFILE:
        raise ValueError("watchdog restoration profile differs from the published C46 policy")
    lab.PRODUCTION_CONFIG = policy.RESTORE_PROFILE
    management = ManagementLatch()
    def changed():
        if management.latched and matches(lab.read_hold(), args.window, args.holder, args.owner_token, True):
            hardware_review(args.window, args.holder, args.owner_token, management)
    management.changed = changed
    while True:
        hold = lab.read_hold()
        if not matches(hold, args.window, args.holder, args.owner_token, review=True):
            return 0
        if hold.get("management_loss_latched") or hold.get("holder") != args.holder:
            return 1
        if lab.heartbeat_age(hold) > lab.HEARTBEAT_MAX_AGE:
            stamp = str(time.time_ns())
            out = ROOT / "results/private/lab" / ("c46-watchdog-restore-" + args.window) / stamp
            try:
                return 0 if close_window(args.window, args.holder, args.owner_token, args.commit, out, management, True) else 1
            except WaitForOwnedJob as error:
                print("watchdog waiting for the existing owned job: " + str(error), flush=True)
            except BaseException as error:
                if management.latched and matches(lab.read_hold(), args.window, args.holder, args.owner_token, True):
                    hardware_review(args.window, args.holder, args.owner_token, management)
                print(f"watchdog retained lease: {type(error).__name__}: {error}", flush=True)
                if not management.latched and matches(lab.read_hold(), args.window, args.holder, args.owner_token):
                    update(args.window, args.holder, args.owner_token,
                           lambda current: current.update(holder="candidate-round-restoration-review",
                               rule="C46 restoration failed; retain its exact evidence and do not repeat the launch", restoration_error=str(error)))
                return 1
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["watchdog"])
    for key in ("window", "holder", "owner-token", "commit", "restore-profile"):
        parser.add_argument("--" + key, required=True)
    raise SystemExit(watchdog_main(parser.parse_args()))
