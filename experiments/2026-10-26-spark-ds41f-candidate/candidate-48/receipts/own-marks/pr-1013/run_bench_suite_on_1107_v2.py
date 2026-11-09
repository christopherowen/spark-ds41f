"""Run PR #1013 bench_suite.py on an image whose SchedulerConfig predates 0766 (shim v2).

Adaptation (ours, not the PR's): spark-ds41f:1107 has no prefill_long_threshold /
max_long_prefills fields, so bench_suite.py's make_sched() raises before any result
is printed. This shim drops the two kwargs and logs every call to stderr. Arms that
pass the defaults (0 / 1) are unaffected; arms that pass prefill_long_threshold != 0
(the two "Size-Classed QoS" arms per length) run size-blind on 1107 and are NOT valid
comparisons. Everything else is bench_suite.py unchanged.

v1 replaced vllm.config.SchedulerConfig with a plain function, which broke class-attribute
access during `import vllm.v1.core.sched.scheduler`. v2 imports the scheduler modules first
and installs a callable that forwards attribute access to the real class.

Usage: python3 run_bench_suite_on_1107_v2.py <bench_suite.py> <boot label>
"""
import runpy
import sys

import vllm.config as vc
import vllm.v1.core.kv_cache_utils  # noqa: F401
import vllm.v1.core.sched.scheduler  # noqa: F401
import vllm.v1.core.single_type_kv_cache_manager  # noqa: F401
import vllm.v1.kv_cache_interface  # noqa: F401
import vllm.v1.outputs  # noqa: F401
import vllm.v1.request  # noqa: F401
import vllm.v1.structured_output  # noqa: F401

_SchedulerConfig = vc.SchedulerConfig


class _SchedulerConfigShim:
    def __call__(self, *args, prefill_long_threshold=0, max_long_prefills=1, **kwargs):
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

    def __getattr__(self, name):
        return getattr(_SchedulerConfig, name)


vc.SchedulerConfig = _SchedulerConfigShim()
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
