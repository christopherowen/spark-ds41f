#!/usr/bin/env python3
"""Own-mark harness for PR #968 (ours, not the PR's): cross-arm output identity of the PR screen's
sink_draft_zero sweep. The PR screen prints only all_bit_exact (adaptive off vs on, within one image) and
row totals; here every configuration's emitted tokens, sink metrics and per-step scheduled rows are kept,
so the base and candidate arms can be compared token for token.

Same loop as screen_remaining_token_cap.py's case 4 (async off/on x max_tokens 1,2,3,4,5,6,8,12,16,32 x
adaptive off/on, tests.v1.core.test_sink_draft_zero's _scheduler(async, 5), SAY_HI script, the same
VLLM_SINK_* environment).
usage: h968.py LABEL  (writes /out/h968-LABEL.json)
"""

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, "/opt/spark-ds41f/candidate/vllm")
LABEL = sys.argv[1]


def main():
    import vllm.v1.core.sched.scheduler as sched_mod
    from tests.v1.core import test_sink_draft_zero as tsdz

    src = open(sched_mod.__file__).read()
    os.environ["VLLM_SINK_DETECTOR_COLUMN"] = "2455"
    os.environ["VLLM_SINK_DETECTOR_THRESHOLD"] = "85000"
    os.environ["VLLM_SINK_REJECTION"] = "1"
    os.environ["VLLM_SINK_REJECTION_MAX_RETRIES"] = "3"
    out = {"label": LABEL, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scheduler_py_sha256": hashlib.sha256(src.encode()).hexdigest()[:12],
           "scheduler_has_max_decode_new_tokens": "_max_decode_new_tokens" in src, "configs": []}
    for async_sched in (False, True):
        for max_toks in (1, 2, 3, 4, 5, 6, 8, 12, 16, 32):
            for adaptive_on in (False, True):
                sched = tsdz._scheduler(async_sched, 5)
                sched.enable_adaptive_verification = adaptive_on
                req = tsdz._make_request("a", max_tokens=max_toks)
                sched.add_request(req)
                step_rows = []
                worker = tsdz.ScriptedWorker(sched, {"a": tsdz.SAY_HI}, 5)
                run = tsdz.drive(sched, worker, async_scheduling=async_sched,
                                 on_schedule=lambda so, sr=step_rows: sr.append(so.total_num_scheduled_tokens))
                emitted = list(run.emitted["a"])
                metrics = run.finished["a"].sink_metrics
                out["configs"].append({
                    "async": async_sched, "max_tokens": max_toks, "adaptive": adaptive_on,
                    "emitted": emitted, "emitted_len": len(emitted),
                    "sink_metrics": repr(metrics), "step_rows": step_rows, "rows": sum(step_rows)})
    digest = hashlib.sha256(json.dumps([[c["async"], c["max_tokens"], c["adaptive"], c["emitted"], c["sink_metrics"]]
                                        for c in out["configs"]]).encode()).hexdigest()[:16]
    out["emitted_and_metrics_digest"] = digest
    out["emitted_only_digest"] = hashlib.sha256(json.dumps([c["emitted"] for c in out["configs"]]).encode()).hexdigest()[:16]
    out["total_rows_adaptive_off"] = sum(c["rows"] for c in out["configs"] if not c["adaptive"])
    out["total_rows_adaptive_on"] = sum(c["rows"] for c in out["configs"] if c["adaptive"])
    out["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h968-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k != "configs"}))


if __name__ == "__main__":
    main()
