"""Run PR #1013 bench_suite.py on an image whose SchedulerConfig predates 0766.

Adaptation (ours, not the PR's): spark-ds41f:1107 has no prefill_long_threshold /
max_long_prefills fields, so bench_suite.py's make_sched() raises before any result
is printed. This shim drops the two kwargs and logs every call to stderr. Arms that
pass the defaults (0 / 1) are unaffected; arms that pass prefill_long_threshold != 0
(the two "Size-Classed QoS" arms per length) run size-blind on 1107 and are NOT valid
comparisons. Everything else is bench_suite.py unchanged.

Usage: python3 run_bench_suite_on_1107.py <bench_suite.py> <boot label>
"""
import runpy
import sys

import vllm.config as vc

_SchedulerConfig = vc.SchedulerConfig


def SchedulerConfig(*args, prefill_long_threshold=0, max_long_prefills=1, **kwargs):
    valid = prefill_long_threshold == 0
    note = "(defaults, valid)" if valid else "(DROPPED, arm invalid on this image)"
    print(
        "SHIM SchedulerConfig"
        f" prefill_long_threshold={prefill_long_threshold}"
        f" max_long_prefills={max_long_prefills}"
        f" max_parallel_prefills={kwargs.get('max_parallel_prefills')}"
        f" align={kwargs.get('align_prefill_chunks')}"
        f" yield={kwargs.get('prefill_yield')} {note}",
        file=sys.stderr,
        flush=True,
    )
    return _SchedulerConfig(*args, **kwargs)


vc.SchedulerConfig = SchedulerConfig
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
