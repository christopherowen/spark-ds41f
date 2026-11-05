#!/usr/bin/env python3
"""Wait locally for the exact queued request; never acquire or change a window."""
import json
import subprocess
import sys
import time


PROGRAM = """
import datetime,json,pathlib
h=pathlib.Path.home()
def read(name):
 p=h/name;return json.loads(p.read_text()) if p.exists() else None
print(json.dumps({'observed':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'hold':read('spark-hold.json'),'request':read('spark-request.json'),'legacy':read('spark3-request.json')}))
"""


def main():
    previous = None
    acknowledged = False
    while True:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx1", "python3", "-"],
                                input=PROGRAM, text=True, capture_output=True, timeout=25)
        if result.returncode:
            print(json.dumps({"operator_observer_exit": result.returncode, "stderr": result.stderr}), flush=True)
            return 2
        state = json.loads(result.stdout)
        hold, request = state["hold"], state["request"]
        ack = (hold or {}).get("queued_after_current_cohort", {})
        if (ack.get("who") == "copy-default-screening" and
                ack.get("request_sha256") == "a6b4d01048300d742325f5f6403c3f4ffdbbff0e89fd1ed7c1b1d9e4755ead5c"):
            acknowledged = True
        compact = {"holder": (hold or {}).get("holder"), "window": (hold or {}).get("window"),
                   "current": (hold or {}).get("current"), "request": request, "legacy": state["legacy"],
                   "exact_request_acknowledged": acknowledged}
        if compact != previous:
            print(json.dumps(state), flush=True)
            previous = compact
        if hold and "hardware-review" in hold.get("holder", "").lower():
            return 3
        matching = bool(request and request.get("who") == "copy-default-screening" and request.get("candidate") == 42
                        and request.get("source_commit") == "a1690f3e3d289eb034c0ef3a6e9cf0da51c0cee1")
        if not matching and not (request is None and acknowledged):
            print(json.dumps({"review_required": "queued request changed or disappeared; do not overwrite or launch"}), flush=True)
            return 4
        if hold is None and state["legacy"] is None:
            print(json.dumps({"ready_for_fresh_guarded_review": True, "state": state}), flush=True)
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
