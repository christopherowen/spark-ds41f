#!/usr/bin/env python3
"""Refresh an open lab window's heartbeat while its holder works step by step.

`scripts/lab.py run` refreshes `heartbeat` in `~/spark-hold.json` every minute; work run
by hand inside a window (here: a sync to an experiment commit, an arm's boot and a bench)
must do the same, or the watchdog restores production after 15 minutes. Every 60 s this
rewrites the heartbeat (and `current`), atomically, while the hold is a lab window's and
until STOP_FILE exists. It never creates a hold.

usage: hold_heartbeat.py NOTE [STOP_FILE]
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

HOLD = Path.home() / "spark-hold.json"
HOLDER_PREFIX = "spark-lab"


def main() -> int:
    note = sys.argv[1] if len(sys.argv) > 1 else "manual steps"
    stop = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "spark-heartbeat.stop"
    while not stop.exists():
        try:
            hold = json.loads(HOLD.read_text())
        except (OSError, ValueError):
            print("no hold file; stopping", flush=True)
            return 1
        if not str(hold.get("holder", "")).startswith(HOLDER_PREFIX):
            print(f"hold belongs to {hold.get('holder')!r}; stopping", flush=True)
            return 1
        hold["heartbeat"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hold["current"] = note
        tmp = HOLD.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(hold, indent=2) + "\n")
        os.replace(tmp, HOLD)
        print(f"{hold['heartbeat']} heartbeat", flush=True)
        time.sleep(60)
    print("stop file present; stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
