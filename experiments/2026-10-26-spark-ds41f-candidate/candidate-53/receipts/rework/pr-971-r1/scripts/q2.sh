#!/bin/bash
# q2.sh STEP: the #971 round-1 queue on sp8 after q.sh (stopped at 20:10Z while prime-r1 waited for the
# lock; it had run import-{base,r1,orig} and prime-base). Each step takes /tmp/spark-lab-gpu.lock ONCE
# and runs its jobs back to back inside it (one fresh container each, through run.sh, which still
# refuses on a hold file or another running container). A refused batch is retried every 60 s for up
# to 60 min, skipping the jobs that already ran; a hold file stops the queue.
#   prime:  prime-r1, prime-orig (TileLang JIT caches; not analysed), import2-{base,r1,orig}
#   script: the PR's own screen on r1, boot1 and boot2
#   ab:     base b1, r1 b1, orig b1, orig b2, r1 b2, base b2 (ABBA for base/r1 and r1/orig)
#   pytest: the PR's tests on r1, twice
B=$HOME/spark-lab/agents/candidate-53-rework-971; cd $B
H=$B/harness
T="-e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2"
M="-v $H:/h:ro"
S="-v $B/pr-971-src/screen_copy_drafter_smooth.py:/tmp/screen_copy_drafter_smooth.py:ro"
mkdir -p out/.done
r() {  # inside a batch: skip a job that ran; a refusal (3) ends the batch for a retry
  [ -e out/.done/$1 ] && return 0
  ./run.sh "$@"; rc=$?
  [ $rc -eq 3 ] && exit 3
  touch out/.done/$1
  return $rc
}
inner() {
  case $1 in
    prime)
      for arm in r1 orig; do r prime-$arm $arm $M $T -- python3 /h/warm.py /h/ab971r1.py --label prime-$arm --out /out/prime-$arm.jsonl --warm 2 --repeats 5; done
      for arm in base r1 orig; do r import2-$arm $arm $M $T -- python3 /h/import_check.py; done;;
    script)
      r script-r1-boot1 r1 $M $S -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot1
      r script-r1-boot2 r1 $M $S -- python3 /h/warm.py /tmp/screen_copy_drafter_smooth.py --boot boot2;;
    ab)
      for lab in base-boot1:base r1-boot1:r1 orig-boot1:orig orig-boot2:orig r1-boot2:r1 base-boot2:base; do
        l=${lab%%:*}; arm=${lab##*:}
        r ab-$l $arm $M $T -- python3 /h/warm.py /h/ab971r1.py --label $l --out /out/ab-$l.jsonl
      done;;
    pytest)
      for k in 1 2; do r pytest-r1-boot$k r1 -- pytest -v --noconftest tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py tests/v1/spec_decode/test_adaptive_verification.py tests/v1/worker/test_prepare_kernels.py; done;;
  esac
}
if [ "$1" = inner ]; then inner $2; exit 0; fi
for step in "$@"; do
  for i in $(seq 1 60); do
    [ -e $HOME/spark-hold.json ] && { echo "HOLD present; queue stops at $step"; exit 9; }
    echo "batch $step: waiting for the lock $(date -u +%FT%TZ)"
    flock /tmp/spark-lab-gpu.lock env SPARK_LAB_GPU_LOCK_HELD=$step ./q2.sh inner $step; rc=$?
    echo "batch $step: rc=$rc $(date -u +%FT%TZ)"
    [ $rc -ne 3 ] && break
    sleep 60
  done
done
echo "queue $* done"
