"""What sizes run-iket's buffer: the tracker's per-warp maximum against the trace's.

usage: trace_counts.py RUN_DIR [SOURCES_DIR]

Prints the tracker's ``recommended_max_ts_cnt_per_warp`` (``RUN_DIR/tracker/pid_*``,
kept with ``profile --keep``) and, per kernel, the most events one warp recorded in the
trace (two per range, one per marker) and the warps that recorded any. With
SOURCES_DIR, also the IKET mode's static bound on a warp's events for each kernel.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    for tracker in sorted(run.glob("tracker/*/tracker.json")):
        data = json.loads(tracker.read_text())
        print(f"{tracker.parent.name}: recommended_max_ts_cnt_per_warp "
              f"{data.get('recommended_max_ts_cnt_per_warp')}")
    bounds = {}
    if len(sys.argv) > 2:
        for path in Path(sys.argv[2]).glob("*.json"):
            for site in json.loads(path.read_text())["sites"]:
                if "events_per_warp_bound" in site:
                    bound = site["events_per_warp_bound"]
                    previous = bounds.get(site["kernel"], 0)
                    bounds[site["kernel"]] = None if None in (bound, previous) else max(previous, bound)
    trace = max(run.glob("*.trace.json"), key=lambda p: p.stat().st_size)
    data = json.loads(trace.read_text())
    most = defaultdict(int)
    recording = defaultdict(set)
    for launch in data["launches"]:
        events = defaultdict(int)
        for r in launch["ranges"]:
            events[r["warpLocIdxs"][0]] += 2
        for m in launch["markers"]:
            events[m["locIdx"]] += 1
        name = launch["kernelName"]
        if events:
            most[name] = max(most[name], max(events.values()))
            recording[name].add(len(events))
    for name in sorted(most, key=lambda n: -most[n]):
        bound = bounds.get(name, "?")
        print(f"{most[name]:6} events in one warp, {max(recording[name]):6} warps recording, "
              f"bound {bound}: {name}")


if __name__ == "__main__":
    main()
