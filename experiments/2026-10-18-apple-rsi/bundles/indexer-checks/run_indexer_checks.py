"""Apple's DS4.1 sparse-indexer checks against our TileLang indexer (rsi_indexer_shim.py).

The task directory is mounted read-only at /apple. Its grading layout (TESTS_DIR: the
verifier package, the evaluation copy and config.json) is staged in /tmp, then, each in its
own process as the task runs them:

1. the public checking script on the public examples (``sparse_indexer/check.py``);
2. the four held-out correctness families (``infb_verify/sparse_cases.py``): two_level,
   packed, paged, replay_spec;
3. the component measurement on the public measurement setups (``sparse_indexer/measure.py``,
   prefill and decode), which reports times and declares no pass.

Every family must be complete, hold exactly the case ids config.json names, and pass every
case. Prints each case's record and ends with "N passed" or "N failed".
"""
import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

APPLE = Path(os.environ.get("APPLE_TASK", "/apple"))
STAGE = Path("/tmp/rsi-tests")
EVAL = STAGE / "eval" / "deepseek_v41_flash"
FAMILIES = ("two_level", "packed", "paged", "replay_spec")
HERE = Path(__file__).resolve().parent


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(APPLE / "tests" / "infb_verify", STAGE / "infb_verify")
    shutil.copytree(APPLE / "environment" / "eval" / "deepseek_v41_flash", EVAL)
    shutil.copy(APPLE / "tests" / "config.json", STAGE / "config.json")


def child(args):
    """One step in this process, the shim installed."""
    os.environ["TESTS_DIR"] = str(STAGE)
    sys.path.insert(0, str(STAGE))
    sys.path.insert(0, str(HERE))
    import rsi_indexer_shim

    rsi_indexer_shim.install()
    if args.family:
        from infb_verify import sparse_cases

        return sparse_cases.main([args.family, "--out", args.out, "--device", "cuda:0"])
    script = EVAL / "sparse_indexer" / ("check.py" if args.check else "measure.py")
    argv = ["--cases", str(EVAL / "sparse_indexer" / "examples.json"), "--output", args.out]
    if args.measure:
        argv = ["--phase", args.measure, *argv]
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


def run(argv, out_dir, timeout):
    command = [sys.executable, str(Path(__file__).resolve()), *argv]
    try:
        process = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout,
                                 cwd=str(out_dir))
        return process.returncode, process.stdout + process.stderr
    except subprocess.TimeoutExpired as exc:
        return None, f"timed out after {timeout} s\n{exc.stdout or ''}{exc.stderr or ''}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--measure", choices=("prefill", "decode"))
    parser.add_argument("--out")
    parser.add_argument("--out-dir", default="/c/apple-rsi-indexer")
    args = parser.parse_args()
    if args.family or args.check or args.measure:
        return child(args)

    stage()
    config = json.loads((STAGE / "config.json").read_text())["infb"]["sparse_indexer"]
    out_dir = Path(args.out_dir) / os.environ.get("RSI_LABEL", "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = passes = 0

    rc, text = run(["--check", "--out", str(out_dir / "check.json")], out_dir, 1800)
    print(f"== public check (exit {rc})\n{text.strip()}", flush=True)
    if rc == 0:
        passes += 1
    else:
        failures += 1

    for family in FAMILIES:
        path = out_dir / f"cases-{family}.json"
        rc, text = run(["--family", family, "--out", str(path)], out_dir,
                       float(config["operation_caps_s"]["correctness_family"]))
        print(f"== {family} (exit {rc})\n{text.strip()[-4000:]}", flush=True)
        record = json.loads(path.read_text()) if path.exists() else {}
        cases = record.get("cases", [])
        for case in cases:
            print(json.dumps({k: case[k] for k in ("case", "status", "n_problems", "problems", "metrics", "seconds")
                              if k in case}), flush=True)
        ids = sorted(c["case"] for c in cases)
        expected = sorted(config["case_ids"][family])
        complete = record.get("complete") and ids == expected
        if not complete:
            print(f"{family}: INCOMPLETE (complete={record.get('complete')}, cases {ids} vs {expected})", flush=True)
            failures += 1
        bad = [c for c in cases if c.get("status") != "pass"]
        failures += len(bad)
        passes += len(cases) - len(bad)

    for phase in ("prefill", "decode"):
        rc, text = run(["--measure", phase, "--out", str(out_dir / f"measure-{phase}.json")], out_dir, 1800)
        print(f"== measure {phase} (exit {rc}; times only, no bar)\n{text.strip()[-3000:]}", flush=True)
        if rc != 0:
            failures += 1

    print(f"{passes} passed" if not failures else f"{failures} failed, {passes} passed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
