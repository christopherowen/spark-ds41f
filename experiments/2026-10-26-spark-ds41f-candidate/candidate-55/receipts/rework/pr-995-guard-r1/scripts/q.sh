#!/bin/bash
# q.sh: the queue as run (sp8). Each step is one fresh container (a boot) through run.sh, which
# takes /tmp/spark-lab-gpu.lock and refuses (exit 3) while ~/spark-hold.json exists or any
# container runs; a refused step is retried every 60 s, up to 30 times.
# Bench, ABBA in one session: base boot1, cand boot1, cand boot2, base boot2. A boot runs the
# harness (warm.py + guard995.py) and then benchmark_copy_drafts.py --gate --cases copy, the
# copy-step bench of #977's own mark (candidate-53 receipts), unmodified, as #977's ran it.
# Then the tests: the PR's pytest command on the candidate arm twice; on the base arm, the
# canary test from the candidate's test file against candidate-54's kernels, and the base's
# own suite once as the reference.
W=$HOME/spark-lab/agents/round3-995-guard; cd $W
retry() { for i in $(seq 1 30); do [ -e $HOME/spark-hold.json ] && { echo "hold file present: stop"; exit 9; }; ./run.sh "$@"; rc=$?; [ $rc -ne 3 ] && return $rc; sleep 60; done; return 3; }
PYT="-o addopts= --noconftest -p no:cacheprovider -v"
for spec in "base 1" "cand 1" "cand 2" "base 2"; do
  set -- $spec
  retry bench-$1-boot$2 $1 -- python3 /h/warm.py /h/guard995.py --label $1-boot$2 --out /out/bench-$1-boot$2.jsonl
  retry gate-$1-boot$2 $1 -- python3 benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy
done
retry pytest-cand-boot1 cand -- python3 -m pytest $PYT tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py
retry canary-base-boot1 base -v $W/ovl/cand/tests/v1/spec_decode/test_copy_drafts.py:/tmp/canary/test_copy_drafts.py:ro -- python3 -m pytest $PYT /tmp/canary/test_copy_drafts.py -k test_negative_slot_bounds_guard_in_alloc_and_apply
retry pytest-cand-boot2 cand -- python3 -m pytest $PYT tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py
retry pytest-base-boot1 base -- python3 -m pytest $PYT tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py
echo "QUEUE DONE $(date -u +%FT%TZ)"
