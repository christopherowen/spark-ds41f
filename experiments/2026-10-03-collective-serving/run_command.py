"""Run one existing cluster/bench command while refreshing the owned window.

Usage: python3 run_command.py LOGFILE -- COMMAND [ARG ...]
Does not acquire the window or change serving configuration on its own.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

HOLDER = "collective-policy-serving"


def heartbeat() -> None:
    code = (
        "import pathlib,json,datetime;"
        "p=pathlib.Path.home()/'spark3-hold.json';d=json.loads(p.read_text());"
        f"assert d['holder']=={HOLDER!r};"
        "d['heartbeat']=datetime.datetime.now(datetime.timezone.utc).isoformat();"
        "p.write_text(json.dumps(d,indent=2)+'\\n')"
    )
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "swank@dgx1", shlex.join(["python3", "-c", code])],
        check=True, timeout=15,
    )


if __name__ == "__main__":
    log_path = Path(sys.argv[1])
    command = sys.argv[sys.argv.index("--") + 1:]
    if not command:
        raise SystemExit("command is required")
    heartbeat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while True:
                try:
                    result = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    heartbeat()
        except BaseException:
            # Let the coordinated cluster command perform its own cleanup.
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
            raise
        finally:
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                           exit=process.poll())
            log_path.with_suffix(log_path.suffix + ".json").write_text(
                json.dumps(receipt, indent=2) + "\n")
    print(f"exit={result}; log={log_path}", flush=True)
    raise SystemExit(result)
