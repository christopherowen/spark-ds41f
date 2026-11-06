"""15-minute stale-heartbeat watchdog for dgx4-verification-shapes-20261106."""
import datetime
import json
import os
import subprocess
import time

HOLD_PATH = os.path.expanduser("~/spark-hold.json")
TOKEN = "dgx4-verification-shapes-20261106"

while True:
    time.sleep(30)
    if not os.path.exists(HOLD_PATH):
        break
    try:
        with open(HOLD_PATH) as f:
            h = json.load(f)
    except Exception:
        continue
    if h.get("token") != TOKEN:
        break
    try:
        hb = datetime.datetime.strptime(h["heartbeat"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
        age = (datetime.datetime.now(datetime.timezone.utc) - hb).total_seconds()
    except Exception:
        continue
    if age > 900:
        for name in (
            f"verification-shapes-gate-{TOKEN}",
            "verification-shapes-bench-boot-1",
            "verification-shapes-bench-boot-2",
        ):
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            with open(HOLD_PATH) as f:
                h2 = json.load(f)
            if h2.get("token") == TOKEN:
                os.unlink(HOLD_PATH)
        except Exception:
            pass
        break
