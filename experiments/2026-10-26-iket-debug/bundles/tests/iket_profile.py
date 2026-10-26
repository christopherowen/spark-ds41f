"""run-iket profile, with each warp's record sized from every process's dry run.

usage: iket_profile.py OUT_DIR -- COMMAND...

run-iket sizes every warp's record from its dry run's recommended_max_ts_cnt_per_warp,
but it reads only the first process directory it lists (``iket.cli.run_iket``, the loop
that breaks at the first count). A child process that loads CUDA and launches nothing
records 0, and when its directory comes first every warp gets a 16-byte header and no
slots: the exit dump then crashes in ``iket::to_json`` (run 28 on dgx3, whose harness
started two such children).

So the dry run runs alone (``--skip-run``) into OUT_DIR/dry; the largest count over its
processes sizes the measured run, which reuses that tracker (``--use-tracker-dir``) and
writes OUT_DIR/trace. Its records stay as run-iket decodes them,
OUT_DIR/trace/iket/pid_*/iket.decoded_results.json, which is the format of the json
postprocess's ``*.trace.json`` (that postprocess copies the file). The perfetto and html
postprocesses, built in memory, took more than 100 GB for a decode-step run's 2.6 GB of
records (run 29); ``run-iket postprocess`` makes them for a smaller run.
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    command = sys.argv[sys.argv.index("--") + 1:]
    dry = subprocess.run(["run-iket", "--output-dir", str(out / "dry"), "--skip-run", "profile",
                          "--keep", "--postprocess", "none", "--", *command])
    if dry.returncode:
        sys.exit(dry.returncode)
    counts = {path.parent.name: json.loads(path.read_text()).get("recommended_max_ts_cnt_per_warp", 0)
              for path in sorted((out / "dry" / "tracker").glob("pid_*/tracker.json"))}
    print(f"dry run's per-warp counts by process: {counts}", flush=True)
    most = max(counts.values(), default=0)
    if not most:
        raise SystemExit("no process of the dry run recorded a count")
    sys.exit(subprocess.run(["run-iket", "--output-dir", str(out / "trace"), "profile", "--keep",
                             "--use-tracker-dir", str(out / "dry" / "tracker"),
                             "--max-ts-cnt-per-warp", str(most), "--postprocess", "none", "--",
                             *command]).returncode)


if __name__ == "__main__":
    main()
