"""Draft in-process builder capability: exact lease, held locks and active guards.

Not a scheduler. Only the C45 bounded builder constructs this while it owns the
real local locks and has connected its shared-window/participating-claim probes.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


class BuildControl:
    def __init__(self, *, commit, window, holder, token, locks, watch, scope_check,
                 management, receipt, output, save):
        self.commit, self.window, self.holder, self.token = commit, window, holder, token
        self.locks, self.watch, self.scope_check = locks, watch, scope_check
        self.management, self.receipt, self.output, self.save = management, receipt, Path(output), save
        self.process = None
        self.containers = {}
        self.halted = []
        self.active = True

    def assert_scope(self, authority=None):
        if not self.active or self.management.latched or self.halted:
            raise RuntimeError("C45 builder guard is inactive or latched")
        if set(self.locks) != {"gpu", "cpu-work", "cpu-offline"}:
            raise RuntimeError("all actual C45 CPU/GPU lock handles are mandatory")
        observations = {}
        for name, handle in self.locks.items():
            if handle.closed:
                raise RuntimeError("C45 lock closed: " + name)
            actual, path = os.fstat(handle.fileno()), Path(handle.name).stat()
            if (actual.st_dev, actual.st_ino) != (path.st_dev, path.st_ino):
                raise RuntimeError("C45 lock inode changed: " + name)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            observations[name] = {"path": str(handle.name), "device": actual.st_dev, "inode": actual.st_ino}
        state = self.scope_check()
        wanted = {"commit": self.commit, "window": self.window, "holder": self.holder, "owner_token": self.token}
        if not isinstance(state, dict) or state.get("passed") is not True or any(state.get(k) != v for k, v in wanted.items()):
            raise RuntimeError("actual C45 shared lease/participating claims do not match")
        if authority is not None and any(authority.get(k) != v for k, v in wanted.items()):
            raise RuntimeError("feature authority belongs to another execution/lease")
        self.watch.check()
        self.receipt["live_build_scope"] = {**wanted, "held_locks": observations, "claims": state}
        self.save()
        return self.receipt["live_build_scope"]

    def register_container(self, name, token, image):
        self.assert_scope()
        if (not re.fullmatch(r"compiler-c45-(?:gate|initcheck|identity|features-[12])", name)
                or not token or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)):
            raise ValueError("only exact C45 new-image gate ownership may be registered")
        old = self.containers.get(name)
        if old is not None and old != {"token": token, "image": image}:
            raise RuntimeError("refusing to replace another container ownership record")
        self.containers[name] = {"token": token, "image": image}
        self.receipt["owned_gate_containers"] = dict(self.containers)
        self.save()

    def metadata_run(self, argv, **kwargs):
        kwargs.setdefault("timeout", 20)
        try:
            return subprocess.run(argv, **kwargs)
        except subprocess.TimeoutExpired:
            self.management.latch("docker_management_timeout", "owned-build-cleanup", "dgx4", repr(argv))
            raise

    def stop_owned_containers(self):
        results = []
        for name, owned in self.containers.items():
            inspected = self.metadata_run(["docker", "inspect", name], text=True, capture_output=True)
            row = {"name": name, "inspect_exit": inspected.returncode, "stdout": inspected.stdout, "stderr": inspected.stderr}
            results.append(row)
            if inspected.returncode:
                if inspected.returncode == 1 and "no such" in (inspected.stdout + inspected.stderr).lower():
                    continue
                self.management.latch("docker_management_loss", "owned-build-stop", "dgx4", inspected.stderr)
                continue
            state = json.loads(inspected.stdout)[0]
            if (state["Image"] != owned["image"]
                    or state["Config"].get("Labels", {}).get("spark.qualification-token") != owned["token"]):
                row["foreign_state_preserved"] = True
                self.management.latch("owned_container_identity_loss", "owned-build-stop", "dgx4", name)
                continue
            stopped = self.metadata_run(["docker", "stop", "--time", "15", state["Id"]], text=True, capture_output=True)
            row.update(container_id=state["Id"], stop_exit=stopped.returncode,
                       stop_stdout=stopped.stdout, stop_stderr=stopped.stderr)
            if stopped.returncode:
                self.management.latch("owned_container_stop_unverified", "owned-build-stop", "dgx4", stopped.stderr)
        self.receipt.setdefault("guard_container_stops", []).append(results)
        self.save()

    def halt(self, reason):
        reason = str(reason)
        if reason not in self.halted:
            self.halted.append(reason)
        conditions = self.receipt.setdefault("conditions", {}).setdefault("halted", [])
        if reason not in conditions:
            conditions.append(reason)
        self.management.observe_text("HALT: " + reason, "builder-guard", "dgx4")
        child = self.process
        if child is not None and child.poll() is None:
            if os.getpgid(child.pid) != child.pid:
                self.management.latch("owned_process_identity_loss", "builder-guard", "dgx4", str(child.pid))
                raise RuntimeError("refuse to signal an unowned process group")
            os.killpg(child.pid, signal.SIGINT)
        self.save()

    def run(self, argv, name, *, stream=None, retain_failed_verdict=False):
        self.assert_scope()
        record = {"argv": list(argv), "started": time.time(), "log": name + ".log"}
        if stream is not None:
            record["log"] = str(Path(stream.name).relative_to(self.output))
        self.receipt.setdefault("steps", []).append(record)
        self.save()
        owned_stream = stream is None
        if owned_stream:
            stream = (self.output / record["log"]).open("x")
        interrupted = None
        try:
            self.process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            record.update(pid=self.process.pid, process_group=self.process.pid, new_session=True)
            self.save()
            next_scope = time.monotonic()
            while self.process.poll() is None:
                try:
                    self.watch.check()
                    if time.monotonic() >= next_scope:
                        self.assert_scope()
                        next_scope = time.monotonic() + 15
                except BaseException as error:
                    self.halt(error)
                if self.halted or self.management.latched:
                    if interrupted is None:
                        interrupted = time.monotonic()
                        self.stop_owned_containers()
                    elapsed = time.monotonic() - interrupted
                    if elapsed > 30 and self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    elif elapsed > 15 and self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGTERM)
                time.sleep(1)
            code = self.process.wait()
            record.update(exit=code, ended=time.time())
            self.save()
        finally:
            if owned_stream:
                stream.close()
        self.management.observe_file(self.output / record["log"], name)
        if self.halted or self.management.latched or (code and not retain_failed_verdict):
            raise RuntimeError(name + " exited " + str(code) + "; " + repr(self.halted))
        return subprocess.CompletedProcess(argv, code)

    def run_container(self, argv, stream, boot, image, token, name):
        self.register_container(name, token, image)
        return self.run(argv, "compiler-features-" + Path(boot).name, stream=stream, retain_failed_verdict=True)
