#!/usr/bin/env bash
# PR #996 rework r1 batch on sp8: every GPU command under flock $L, one fresh container each.
. $HOME/spark-lab/agents/candidate-49-rework-996/shim/env.sh
cd $B
# failing reproduction with the own-marks harness: candidate-49 as built
shim/run.sh infer-candidate49 $C -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996-infer-c49 --entrypoint "" -w $V $H $C python3 /tmp/bench_996_route_local.py infer /out/infer-candidate49.json
for boot in 1 2; do
  # #996's own tests and the SP prefill and router tests it lists, with the new tests
  shim/run.sh pytest-corrected-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996-pyt-b$boot --entrypoint "" -w $V $M $R pytest --noconftest tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py -p no:cacheprovider
  # the own-marks harness's inference-mode cases (5 of 5 raised on candidate-49)
  shim/run.sh infer-corrected-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996-infer-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/bench_996_route_local.py infer /out/infer-corrected-boot$boot.json
  # bit-exactness under inference mode: every rank, TP 3/4, T 1024..16384, 2 steps x 20 layers vs 1107's masks
  shim/run.sh exact-corrected-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996-exact-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/exact_996.py exact /out/exact-corrected-boot$boot.json
  # the route_local bench of the own mark, unchanged (no_grad), against the recorded 1107 and candidate-49 boots
  shim/run.sh bench-corrected-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996-bench-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/bench_996_route_local.py bench /out/route-local-corrected-boot$boot.json
  # the same timing under inference mode, this image's _local_mask against 1107's, interleaved
  shim/run.sh ibench-corrected-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996-ibench-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/exact_996.py ibench /out/ibench-corrected-boot$boot.json
done
echo BATCH_DONE
