#!/bin/bash
# q.sh: the #971 round-1 queue on sp8, one GPU job at a time (each through run.sh). A job refused
# for a running container is retried every 60 s for up to 30 min; a hold file stops the queue.
# Order: an import check per arm, one priming run per arm (TileLang JIT caches; not analysed), the PR's own screen on r1
# twice, then the image A/B in one session: base b1, r1 b1, orig b1, orig b2, r1 b2, base b2
# (ABBA for base/r1 and for r1/orig), then the PR's tests on r1 twice.
B=$HOME/spark-lab/agents/candidate-53-rework-971; cd $B
H=$B/harness
r() {
  for i in $(seq 1 30); do
    [ -e $HOME/spark-hold.json ] && { echo "HOLD present; queue stops at $1"; exit 9; }
    ./run.sh "$@"; rc=$?
    [ $rc -ne 3 ] && return $rc
    sleep 60
  done
}
step=${1:-all}
T="-e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2"
M="-v $H:/h:ro"
S="-v $B/pr-971-src/screen_copy_drafter_smooth.py:/tmp/screen_copy_drafter_smooth.py:ro"
if [ $step = all ] || [ $step = import ]; then
  for arm in base r1 orig; do r import-$arm $arm $T -- python3 -c '
import importlib, inspect
for m in ["vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu.spec_decode.adaptive_verification", "vllm.v1.worker.gpu.spec_decode.tilelang_copy"]:
    mod = importlib.import_module(m); print(m, mod.__file__)
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as a
import vllm.v1.worker.gpu.model_runner as r
print("_DEAD_ROWS_TAU", a._DEAD_ROWS_TAU, "dead_rows_enabled", a.dead_rows_enabled())
print("SINGLE_REQ_TILE_SPILL_SCALE", getattr(a, "SINGLE_REQ_TILE_SPILL_SCALE", None), "_ZERO_CONFIDENCE_LOG_TAU", getattr(a, "_ZERO_CONFIDENCE_LOG_TAU", None))
print("declare_step_kernels: copy drafter declares dead rows:", "dead_rows_enabled() or self.copy_drafter" in inspect.getsource(r.GPUModelRunner.declare_step_kernels))
print("dead_row_args has the zero-confidence floor:", "has_copy_tail" in inspect.getsource(a.AdaptiveVerificationManager.dead_row_args))
print("copy_apply survival:", "dspark_survival" in inspect.getsource(importlib.import_module("vllm.v1.worker.gpu.spec_decode.tilelang_copy").copy_apply_kernel))
'; done
fi
if [ $step = all ] || [ $step = prime ]; then
  for arm in base r1 orig; do r prime-$arm $arm $M $T -- python3 /h/warm.py /h/ab971r1.py --label prime-$arm --out /out/prime-$arm.jsonl --warm 2 --repeats 5; done
fi
if [ $step = all ] || [ $step = script ]; then
  r script-r1-boot1 r1 $M $S -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot1
  r script-r1-boot2 r1 $M $S -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot2
fi
if [ $step = all ] || [ $step = ab ]; then
  for lab in base-boot1:base r1-boot1:r1 orig-boot1:orig orig-boot2:orig r1-boot2:r1 base-boot2:base; do
    l=${lab%%:*}; arm=${lab##*:}
    r ab-$l $arm $M $T -- python3 /h/warm.py /h/ab971r1.py --label $l --out /out/ab-$l.jsonl
  done
fi
if [ $step = all ] || [ $step = pytest ]; then
  for k in 1 2; do r pytest-r1-boot$k r1 -- pytest -v --noconftest tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py tests/v1/spec_decode/test_adaptive_verification.py tests/v1/worker/test_prepare_kernels.py; done
fi
echo "queue $step done"
