#!/usr/bin/env bash
# PR #996 rework r2 batch on sp8: every GPU command under flock $L, one fresh container each.
. $HOME/spark-lab/agents/candidate-49-rework-996-r2/shim/env.sh
cd $B
# round 1 test file unchanged over the round 2 router: which of its assertions encode views being kept
shim/run.sh r1tests-unchanged-on-r2 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-r1tests --entrypoint "" -w $V $M_X $R pytest --noconftest tests/kernels/moe/test_deepseek_v41_tilelang_router.py -rf -p no:cacheprovider
for boot in 1 2; do
  # PR tests, round 1 tests and round 2 tests: SP prefill and router test files
  shim/run.sh pytest-r2-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-pyt-b$boot --entrypoint "" -w $V $M $R pytest --noconftest tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py -p no:cacheprovider
  # own-marks harness inference-mode cases
  shim/run.sh infer-r2-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-infer-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/bench_996_route_local.py infer /out/infer-r2-boot$boot.json
  # bit-exactness under inference mode, every rank, TP 3/4, T 1024..16384, 2 forwards x 20 layers vs 1107 masks
  shim/run.sh exact-r2-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-exact-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/exact_996.py exact /out/exact-r2-boot$boot.json
  # own-mark route_local bench (no_grad), round 2; round 1 tree as an in-session control, ABBA
  if [ $boot = 1 ]; then
    shim/run.sh bench-r2-boot1 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-bench-b1 --entrypoint "" -w $V $M $H $R python3 /tmp/bench_996_route_local.py bench /out/route-local-r2-boot1.json
    shim/run.sh bench-r1ctl-boot1 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-r1ctl-b1 --entrypoint "" -w $V $M_R1 $H $R python3 /tmp/bench_996_route_local.py bench /out/route-local-r1ctl-boot1.json
  else
    shim/run.sh bench-r1ctl-boot2 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-r1ctl-b2 --entrypoint "" -w $V $M_R1 $H $R python3 /tmp/bench_996_route_local.py bench /out/route-local-r1ctl-boot2.json
    shim/run.sh bench-r2-boot2 $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-bench-b2 --entrypoint "" -w $V $M $H $R python3 /tmp/bench_996_route_local.py bench /out/route-local-r2-boot2.json
  fi
  # in-process A/B under inference mode: round 2 _local_mask against 1107 body, interleaved per step
  shim/run.sh ibench-r2-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-ibench-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/exact_996.py ibench /out/ibench-r2-boot$boot.json
  # host cost of one routing layer local masks: 1107, round 1 and round 2 bodies, interleaved
  shim/run.sh cpu-r2-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c49rw996r2-cpu-b$boot --entrypoint "" -w $V $M $H $R python3 /tmp/cpu_996.py /out/cpu-r2-boot$boot.json
done
echo BATCH_DONE
