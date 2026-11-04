#!/usr/bin/env python3
"""Run first/repeat FP8 gates inside an already reserved, authorized window."""

import argparse
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab
from fp8_results import FIXED_EXACT_FLAG, FIXED_SOURCES, summarize, validate

HOLDER = "spark-lab: owner-authorized lossless FP8 decoder measurements"
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True)
    parser.add_argument("--format", choices=("huffman", "fixed-fields"), default="huffman")
    parser.add_argument("--lanes", type=int, choices=(32, 64, 128, 256, 512), nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    fixed = args.format == "fixed-fields"
    allowed = (128, 256, 512) if fixed else (32, 64, 128)
    args.lanes = args.lanes if args.lanes is not None else list(allowed)
    if not set(args.lanes).issubset(allowed):
        parser.error("parallelism does not belong to the chosen format")
    if len(set(args.lanes)) != len(args.lanes):
        parser.error("lane counts must be distinct")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT / "results/private"):
        parser.error("native receipts must stay under results/private")
    spec_root = HERE / "fp8" / "fixed" if fixed else HERE / "fp8"
    validation = {"axis": "threads", "exact_flag": FIXED_EXACT_FLAG,
                  "sources": FIXED_SOURCES} if fixed else {}
    summary_options = {key: value for key, value in validation.items() if key != "sources"}
    specs = [spec_root / f"kernel-{lanes}{suffix}.json"
             for lanes in args.lanes for suffix in ("", "-repeat")]
    if not args.apply:
        print(json.dumps({"window": args.window, "restore": RESTORE,
                          "specs": [str(p.relative_to(ROOT)) for p in specs],
                          "output": str(out), "closes_window_when_finished": True}, indent=2))
        return 0

    def owns():
        hold = lab.read_hold()
        return bool(hold and hold.get("window") == args.window and hold.get("holder") == HOLDER
                    and hold.get("production_config") == RESTORE)

    if os.uname().nodename != "dgx1" or out.exists() or not owns():
        raise SystemExit("wrong host, prior receipts or foreign/mismatched window; no operation")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("checkout is dirty")
    lab.PRODUCTION_CONFIG = RESTORE
    out.mkdir(parents=True)
    base = json.loads((HERE / "base.json").read_text())
    receipt = {"started": lab.now(), "window": args.window, "steps": [], "summaries": [],
               "format": args.format,
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "restored": False, "serving_performance_verified": False}
    done = threading.Event()

    def save():
        lab.write_json_atomic(out / "receipt.json", receipt)

    def heartbeat():
        while not done.wait(30):
            if owns():
                lab.beat("lossless FP8 decoder first/repeat measurements")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    code = 1
    try:
        hold = lab.read_hold()
        closing = lab.window_should_close(hold)
        # A request explicitly queued behind this already-open screen is for
        # after its fixed first/repeat matrix. Older or unrelated requests
        # prevent a new job. Keep the request visible throughout the screen.
        if closing:
            request_path = getattr(lab, "REQUEST", None)
            request = json.loads(request_path.read_text()) if request_path and request_path.exists() else {}
            if (not str(closing).startswith("cluster requested")
                    or request.get("after_window") != args.window or not hold.get("since")
                    or not request.get("since")
                    or datetime.fromisoformat(request["since"].replace("Z", "+00:00")) <
                       datetime.fromisoformat(hold["since"].replace("Z", "+00:00"))):
                raise RuntimeError("window changed or closure requested; preserve completed observations")
            receipt["queued_return_request"] = request
            save()
        for lanes in args.lanes:
            pair = []
            for suffix in ("", "-repeat"):
                if not owns():
                    raise RuntimeError("window changed or closure requested; preserve completed observations")
                spec_path = spec_root / f"kernel-{lanes}{suffix}.json"
                spec = json.loads(spec_path.read_text())
                step = {"spec": str(spec_path.relative_to(ROOT)), "started": lab.now()}
                receipt["steps"].append(step)
                save()
                plan_argv = [sys.executable, "bin/spark", "--cluster-config", RESTORE,
                             "cluster", "stop"]
                plan = subprocess.run(plan_argv, cwd=ROOT, text=True, capture_output=True)
                lab.write_json_atomic(out / (spec["run"] + "-stop-plan.json"),
                                      {"argv": plan_argv, "exit": plan.returncode,
                                       "stdout": plan.stdout, "stderr": plan.stderr})
                print(plan.stdout, flush=True)
                if plan.returncode or "--apply would refuse" in plan.stdout or "shape:   CHANGED" in plan.stdout:
                    raise RuntimeError("coordinated stop plan refused; no operation")
                log_path = out / (spec["run"] + ".log")
                kernel_step = next(s for s in lab.plan(spec) if s["kind"] == "kernel")
                step["kernel_step"] = kernel_step
                step["helper"] = "spark_ds41f.lab.run_kernel_bundles"
                # One screen is one reserved job. Use the standard coordinated
                # stop and kernel backend, with one restoration at its end.
                with log_path.open("x") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        if not lab.stop_cluster():
                            raise RuntimeError("coordinated stop failed")
                        lab.run_kernel_bundles(kernel_step)
                step.update(ended=lab.now())
                save()
                verdicts_path = ROOT / "results/private/lab" / spec["run"] / "verdicts.json"
                verdicts = json.loads(verdicts_path.read_text())
                step["verdicts"] = verdicts
                if verdicts.get("agree") is not True or len(verdicts.get("verdicts", [])) != 1:
                    raise RuntimeError("kernel job failed or lacks an agreed verdict")
                candidate_path = ROOT / spec["jobs"][0]["bundles"][0]["bundle"] / "candidate.json"
                candidate = json.loads(candidate_path.read_text())
                native_path = lab.compile_cache(candidate) / Path(candidate["argv"][-1]).name
                native = json.loads(native_path.read_text())
                lab.write_json_atomic(out / (spec["run"] + ".json"), native)
                validate(native, verdicts["verdicts"][0], lanes=lanes,
                         image=base["image_id"], revision=base["model_revision"], **validation)
                pair.append(native)
                step["qualified_microbenchmark"] = True
                save()
            receipt["summaries"].append(summarize(pair, **summary_options))
            save()
        receipt["completed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        done.set()
        thread.join(timeout=5)
        if owns():
            receipt["restored"] = lab.window_close("FP8 measurements finished or deferred")
        elif lab.read_hold() is None:
            receipt["restored"] = lab.production_live()
        else:
            receipt["restore_deferred"] = "foreign reservation; no operation"
        if not receipt["restored"]:
            code = 1
        receipt["ended"] = lab.now()
        save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
