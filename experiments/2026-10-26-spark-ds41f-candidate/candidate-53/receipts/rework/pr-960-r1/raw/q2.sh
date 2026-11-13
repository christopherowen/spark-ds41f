#!/bin/bash
# q2.sh STEP...: the #960 round-1 queue on sp8. Each step takes /tmp/spark-lab-gpu.lock ONCE and runs
# its jobs back to back inside it (one fresh container each, through run.sh, which still refuses on a
# hold file or another running container). A refused batch is retried every 60 s for up to 60 min,
# skipping the jobs that already ran; a hold file stops the queue.
#   verify: per arm, SHA-256 of every mounted file as the container sees it (no GPU, no network)
#   boot:   harness/bootpath960.py on orig (the boot failure, reproduced) and r1
#   tests:  the new regression test on orig (must fail) and r1; then on r1, twice, the PR's tests,
#           the runner test file and the KV connector tests
#   origtest: the new regression test on origt (the original 0781 runner with r1's test file)
#   script: the PR's own screen_cudagraph_wins.py (PR head 6cfd0bd3, unmodified) on r1, boot1 and boot2
#   ab:     harness/ab960.py (the own-mark image A/B): base b1, r1 b1, r1 b2, base b2
B=$HOME/spark-lab/agents/candidate-53-rework-960; cd $B
H=$B/harness
E=$B/src/pr-960/experiments/2026-11-08-cudagraph-decode-optimizations
M="-v $E:/exp:ro -v $H:/h:ro"
P="-v $B/src/bundle:/bundle:ro -e PYTHONPATH=/bundle"
NEWTEST=tests/v1/worker/test_gpu_model_runner_v2.py::test_full_cudagraph_decode_attn_prep_follows_kv_connector
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
    verify)
      for arm in base r1 orig; do
        case $arm in base) list=no1009-files.txt;; *) list=changed-files-$arm.txt;; esac
        r verify-$arm $arm --network none -v $B/$list:/list:ro -- bash -c 'while read -r f; do echo "$(sha256sum < $f | cut -c1-12) $f"; done < /list'
      done;;
    boot)
      r bootpath-orig orig $M -- python3 /h/bootpath960.py --label orig --out /out/bootpath-orig.jsonl
      r bootpath-r1 r1 $M -- python3 /h/bootpath960.py --label r1 --out /out/bootpath-r1.jsonl;;
    tests)
      r newtest-orig orig $P -- pytest -p tblib_stub -v $NEWTEST
      r newtest-r1 r1 $P -- pytest -p tblib_stub -v $NEWTEST
      for k in 1 2; do r pytest-r1-boot$k r1 $P -- pytest -p tblib_stub -v tests/v1/attention/test_b12x_v41_ced_metadata.py tests/v1/cudagraph/test_cudagraph_manager.py tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_kv_connector.py; done;;
    origtest)
      r newtest-orig-runner origt $P -- pytest -p tblib_stub -v $NEWTEST;;
    script)
      for k in 1 2; do r script-r1-boot$k r1 $M -- python3 /h/warm.py /exp/screen_cudagraph_wins.py --boot $k --output /out/script-r1-boot$k.json; done;;
    ab)
      for lab in base-boot1:base r1-boot1:r1 r1-boot2:r1 base-boot2:base; do
        l=${lab%%:*}; arm=${lab##*:}
        r ab-$l $arm $M -- python3 /h/ab960.py --label $l --out /out/ab-$l.jsonl
      done;;
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
