#!/usr/bin/env python3
"""Run an unchanged C45 lab spec in its exact lease; the parent alone restores."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab
import c45_policy as policy
import c45_window as window
from c45_management import ManagementLatch


class FunctionalGuard:
    """Same strict watcher/floor/cadence across functional boots and extras only."""
    def __init__(self, nodes, floor, management, out, factory=None):
        self.nodes, self.floor, self.management, self.out = nodes, floor, management, Path(out)
        self.factory = factory or lab.spark.MemoryWatch
        self.watch = None; self.thread = None
        self.reasons = []; self.lock = threading.Lock(); self.stop = threading.Event()
        self.armed = False; self.interrupted = False; self.pid = None; self.pgid = None

    def halt(self, reason):
        with self.lock:
            if str(reason) in self.reasons:
                return
            self.reasons.append(str(reason))
            interrupt = self.armed and not self.interrupted
            if interrupt:
                self.interrupted = True
        print("HALT: " + str(reason), flush=True)
        self.management.observe_text("HALT: " + str(reason), "functional-guard")
        lab.write_json_atomic(self.out / "functional-guard-halt.json", {"halted": self.reasons,
            "pid": self.pid, "pgid": self.pgid, "strict_telemetry": True, "memory_floor_gib": self.floor})
        if interrupt:
            if self.pgid != self.pid or os.getpgid(self.pid) != self.pgid:
                raise RuntimeError("functional guard refuses to signal a foreign process group")
            os.killpg(self.pgid, signal.SIGINT)

    def start(self):
        self.pid = os.getpid(); self.pgid = os.getpgid(self.pid)
        if self.pgid != self.pid:
            raise RuntimeError("functional phase must own its new process group before any watcher or boot")
        self.watch = self.factory(self.nodes, self.floor, self.halt)
        try:
            self.watch.wait_ready()
        except Exception as error:
            self.halt(error)
            raise
        if self.reasons:
            raise RuntimeError("functional guard halted before any boot")
        self.armed = True
        def monitor():
            while not self.stop.wait(1):
                try:
                    self.watch.check()
                except Exception as error:
                    self.halt(error)
        self.thread = threading.Thread(target=monitor, daemon=True); self.thread.start()

    def close(self):
        self.armed = False; self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        result = {"halted": self.reasons, "strict_telemetry": True, "memory_floor_gib": self.floor,
                  "nodes": [node["name"] for node in self.nodes["nodes"]], "pid": self.pid, "pgid": self.pgid}
        if self.watch is not None:
            result.update(thermal=self.watch.thermal_summary(), memory=self.watch.close())
        lab.write_json_atomic(self.out / "functional-guard.json", result)
        return result


def run_phase(spec, args, out):
    management = ManagementLatch()
    original_cli, original_script = lab.spark_cli, lab.run_script
    original_ssh = lab.spark.run_ssh
    bindings = {key: getattr(lab, key) for key in ("PRODUCTION_CONFIG", "hold_is_ours", "window_open", "window_close", "beat", "spark_cli", "run_script")}
    prior_strict = os.environ.get("SPARK_DS41F_STRICT_TELEMETRY")
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    def changed():
        lab.write_json_atomic(out / "management.json", management.snapshot())
        if management.latched and window.matches(lab.read_hold(), args.window, args.holder, args.owner_token, True):
            window.hardware_review(args.window, args.holder, args.owner_token, management)
    management.changed = changed
    def require_owned():
        policy.assert_owned(lab.read_hold(), args.window, args.holder, args.owner_token)
    def defer_close(reason, dry=False):
        lab.write_json_atomic(out / "deferred-restoration.json", {"reason": reason, "parent_owns_restoration": True})
        return False
    def guarded_cli(*argv, dry=False, capture=None):
        require_owned()
        argv = list(argv)
        # Generic lab.boot supplies --change-shape. C45 has no shape change;
        # read the named live/shape plan and execute without that bypass.
        argv = [value for value in argv if value != "--change-shape"]
        if "cluster" in argv and "start" in argv and "--apply" in argv:
            lines = []
            code = original_cli(*[value for value in argv if value != "--apply"], capture=lines)
            plan = "\n".join(lines)
            lab.write_json_atomic(out / f"start-plan-{len(list(out.glob('start-plan-*.json')))}.json",
                                  {"argv": argv, "exit": code, "stdout": plan})
            management.observe_text(plan, "boot-plan")
            if code or management.latched or "--apply would refuse" in plan or "shape:   CHANGES" in plan or "live:" not in plan or "shape:" not in plan:
                raise RuntimeError("named C45 start plan refused; no shape bypass")
        lines = []
        code = original_cli(*argv, dry=dry, capture=lines)
        if capture is not None:
            capture.extend(lines)
        management.observe_text("\n".join(lines), "lab-cli")
        if "bench" in argv and "--suite" in argv and "--output" in argv:
            path = ROOT / argv[argv.index("--output") + 1] / "conditions.json"
            if path.exists():
                conditions = json.loads(path.read_text())
                management.observe_text("HALT: " + str(conditions.get("halted") or ""), "suite-guard")
                wanted = set(policy.NODES[:3])
                thermal = conditions.get("thermal", {})
                if code == 0 and (set(thermal) != wanted or set(conditions.get("memory", {})) != wanted
                        or any(row.get(field) is None for row in thermal.values()
                               for field in ("max_gpu_c", "min_headroom_c", "max_system_c"))):
                    management.latch("guard_telemetry_loss", "suite-guard", detail="required all-rank guard telemetry is incomplete")
            elif code == 0:
                management.latch("guard_telemetry_loss", "suite-guard", detail="successful suite lacks its guard receipt")
        if management.latched:
            raise RuntimeError("C45 hardware/telemetry loss latched; restoration is deferred without another launch")
        return code
    def guarded_script(step):
        require_owned()
        code = original_script(step)
        if Path(step["argv"][0]).name == "collect_runtime.py":
            text = (ROOT / step["out"]).read_text(errors="replace")
            management.observe_text(text, "runtime-metadata")
            if code and re.search(r"ssh.{0,300}(?:exit(?:ed)?[=: ]+255|timed out)", text, re.I | re.S):
                management.latch("management_transport_loss", "runtime-metadata", detail="failed native runtime collector records SSH255/timeout")
        if management.latched:
            raise RuntimeError("runtime management loss; no subsequent boot or restoration launch")
        return code
    def no_open(*_args, **_kwargs):
        raise RuntimeError("C45 child phase cannot acquire another window")
    def guarded_ssh(nodes, node, *argv):
        require_owned()
        try:
            output = management.ssh(lab.spark.ssh_target(nodes, node), list(argv), 'lab-node-command', record_output=False)
        except subprocess.CalledProcessError as error:
            if management.latched:
                raise
            return subprocess.CompletedProcess(argv, error.returncode, error.output, error.stderr)
        return subprocess.CompletedProcess(argv, 0, output, management.reads[-1]['stderr'])
    lab.PRODUCTION_CONFIG = policy.RESTORE_PROFILE
    os.environ["SPARK_DS41F_STRICT_TELEMETRY"] = "1"
    lab.hold_is_ours = lambda hold: window.matches(hold, args.window, args.holder, args.owner_token)
    lab.window_open = no_open
    lab.window_close = defer_close
    # Child progress cannot keep a dead parent controller's lease fresh.
    lab.beat = lambda note=None: require_owned()
    lab.spark_cli, lab.run_script = guarded_cli, guarded_script
    lab.spark.run_ssh = guarded_ssh
    functional_guard = None
    code = 1
    try:
        require_owned()
        if spec.get("run") == "candidate-45-functional-tp3":
            cluster, nodes, _ = lab.production_configuration()
            functional_guard = FunctionalGuard(nodes, float(cluster["host"]["memguard_gib"]) + 1, management, out)
            functional_guard.start()
        code = lab.execute(spec, dry=False, keep_open=True, close_on_failure=False)
    finally:
        try:
            if functional_guard is not None:
                functional_guard.close()
            changed()
        finally:
            for key, value in bindings.items():
                setattr(lab, key, value)
            lab.spark.run_ssh = original_ssh
            if prior_strict is None:
                os.environ.pop("SPARK_DS41F_STRICT_TELEMETRY", None)
            else:
                os.environ["SPARK_DS41F_STRICT_TELEMETRY"] = prior_strict
    return (code or 1) if functional_guard is not None and functional_guard.reasons else code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", type=Path, required=True)
    for key in ("window", "holder", "owner-token"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    return run_phase(json.loads((ROOT / args.spec).read_text()), args, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
