"""C46-only sticky management/guard loss; ordinary metadata failures stay distinct."""
from pathlib import Path
import hashlib
import re
import shlex
import subprocess


class ManagementLatch:
    def __init__(self, changed=None):
        self.latched = False
        self.events = []
        self.reads = []
        self.changed = changed

    def latch(self, category, phase, host=None, detail=""):
        event = {"category": category, "phase": phase, "host": host, "detail": str(detail)[:2000]}
        self.latched = True
        if event not in self.events:
            self.events.append(event)
        if self.changed:
            self.changed()

    def observe_text(self, text, phase, host=None):
        for line in text.splitlines():
            message = line.strip().lower()
            if re.match(r"(?:error: )?(?:[^: ]+: )?ssh: (?:connect|could not resolve|connection|kex_)", message):
                self.latch("management_transport_loss", phase, host, line)
            elif message.startswith(("kex_exchange_identification:", "client_loop: send disconnect:",
                                     "connection timed out during banner exchange")):
                self.latch("management_transport_loss", phase, host, line)
            elif (message.startswith(("halt:", "error:", "runtimeerror:", "benchabort:", "spark_ds41f.bench.benchabort:"))
                  and any(term in message for term in ("memory telemetry stopped reporting", "memory telemetry did not start", "thermal telemetry"))):
                self.latch("guard_telemetry_loss", phase, host, line)
            elif "ssh failed (" in message and message.startswith("error"):
                self.latch("management_transport_loss", phase, host, line)
            elif (message.startswith(("live:", "--apply would refuse:"))
                  and any(term in message for term in ("cannot reach", "ssh failed", "connection refused",
                                                       "connection timed out", "no route to host"))):
                self.latch("management_transport_loss", phase, host, line)

    def observe_file(self, path, phase):
        with Path(path).open(errors="replace") as stream:
            for line in stream:
                self.observe_text(line, phase)

    def observe_conditions(self, conditions, phase, host=None):
        reasons = conditions.get("halted", [])
        if not isinstance(reasons, list):
            reasons = [reasons]
        for reason in reasons:
            if reason:
                self.observe_text("HALT: " + str(reason), phase, host)

    def ssh(self, host, argv, phase, timeout=20, probe=False, record_output=True):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                   "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", host, shlex.join(argv)]
        record = {"host": host, "phase": phase, "argv": command}
        self.reads.append(record)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            record.update(error="TimeoutExpired", stdout=repr(error.stdout), stderr=repr(error.stderr))
            self.latch("management_metadata_timeout", phase, host, error)
            raise
        record.update(exit=result.returncode, stderr=result.stderr)
        if record_output:
            record["stdout"] = result.stdout
        else:
            raw = result.stdout.encode()
            record.update(stdout_bytes=len(raw), stdout_sha256=hashlib.sha256(raw).hexdigest(),
                          output_limit="Original caller retains its native output; raw logs are not duplicated into compact management metadata.")
        if result.returncode == 255 or (probe and result.returncode):
            self.latch("management_transport_loss", phase, host, result.stderr or f"probe exit {result.returncode}")
        self.observe_text(result.stderr, phase, host)
        if self.changed:
            self.changed()
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        return result.stdout

    def snapshot(self):
        return {"management_loss_latched": self.latched, "management_events": list(self.events),
                "management_reads": list(self.reads)}
