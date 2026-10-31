"""run-iket profile, with each warp's record sized from every process's dry run.

usage: iket_profile.py OUT_DIR [--timeout SECONDS] [--log-level LEVEL] -- COMMAND...

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

Each run-iket gets ``--timeout`` seconds (default 1800) in a session of its own, which
is killed whole when they run out: run 30's dry run spun in a graph replay for 45
minutes. Then the end of each process's injection log is printed (``--log-level
debug`` makes it name every launch). ``--dry-only`` stops after the dry run, to repeat
it until it hangs (run 42's TP3 c1 dry run stopped the same way).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def run_iket(argv, timeout):
    """Run ``argv`` in a session of its own; past ``timeout`` seconds, kill the session."""
    process = subprocess.Popen(argv, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print(f"run-iket timed out after {timeout} s; its session was killed", flush=True)
        return 124


def log_tails(directory, lines=40):
    for log in sorted(directory.glob("*/pid_*/smodel.injection.log")):
        text = log.read_text(errors="replace").splitlines()
        print(f"--- {log} (last {min(lines, len(text))} of {len(text)} lines)", flush=True)
        print("\n".join(text[-lines:]), flush=True)


def main():
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--dry-only", action="store_true")
    args = parser.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1:]
    out = Path(args.out)
    common = ["--log-level", args.log_level]
    code = run_iket(["run-iket", "--output-dir", str(out / "dry"), "--skip-run", *common, "profile",
                     "--keep", "--postprocess", "none", "--", *command], args.timeout)
    if code:
        log_tails(out / "dry")
        sys.exit(code)
    counts = {path.parent.name: json.loads(path.read_text()).get("recommended_max_ts_cnt_per_warp", 0)
              for path in sorted((out / "dry" / "tracker").glob("pid_*/tracker.json"))}
    print(f"dry run's per-warp counts by process: {counts}", flush=True)
    if args.dry_only:
        sys.exit(0)
    most = max(counts.values(), default=0)
    if not most:
        raise SystemExit("no process of the dry run recorded a count")
    code = run_iket(["run-iket", "--output-dir", str(out / "trace"), *common, "profile", "--keep",
                     "--use-tracker-dir", str(out / "dry" / "tracker"),
                     "--max-ts-cnt-per-warp", str(most), "--postprocess", "none", "--", *command],
                    args.timeout)
    if code:
        log_tails(out / "trace")
    sys.exit(code)


if __name__ == "__main__":
    main()
