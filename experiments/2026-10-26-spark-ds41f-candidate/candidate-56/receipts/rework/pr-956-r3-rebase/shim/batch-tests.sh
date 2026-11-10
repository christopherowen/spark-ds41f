#!/usr/bin/env bash
# Unit tests on sp8, every GPU command under flock $L, a fresh container each.
# The PR's file list, the rebase's new GPU test, and the files of the base patches the
# rebase meets (#1009 r1's front kernels, #1012 r1's decode overlap in the attention
# tests, #960 r1's CED metadata and runner tests, the CED replay kernels).
. $HOME/spark-lab/agents/round3-956/shim/env.sh
cd $B
FILES="tests/models/test_deepseek_v4_1_compressor_prepare.py tests/models/test_deepseek_v4_1_compressor.py tests/models/test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_ced_model.py tests/models/test_deepseek_v4_1_ced_sp.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/test_deepseek_v41_tilelang_front.py tests/models/test_deepseek_v4_1_prepare.py tests/v1/attention/test_b12x_v41_ced_metadata.py tests/v1/worker/test_gpu_model_runner_v2.py tests/kernels/attention/test_deepseek_v41_ced_replay.py"
for boot in 1 2; do
  shim/run.sh pytest-cand-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name r3-956-pyt-cand-$boot $ENVS --entrypoint "" -w $V $M_CAND $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs $FILES
done
shim/run.sh pytest-base-boot1 $R -- flock $L docker run --rm --gpus all --ipc=host --name r3-956-pyt-base-1 $ENVS --entrypoint "" -w $V $M_BASE $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs $FILES
# The rebase's three test files over the base's code: its new tests must fail there.
M_X=""
for f in $(git -C $B/tree-base diff --name-only $IMG_BASE HEAD) tests/models/test_deepseek_v4_1_ced_model.py tests/models/test_deepseek_v4_1_compressor_prepare.py tests/models/test_deepseek_v4_1_compressor.py; do
  s=$B/tree-base/$f
  case $f in tests/models/test_deepseek_v4_1_ced_model.py|tests/models/test_deepseek_v4_1_compressor_prepare.py|tests/models/test_deepseek_v4_1_compressor.py) s=$B/tree-cand/$f;; esac
  case " $M_X " in *" -v $s:$V/$f:ro "*) continue;; esac
  M_X="$M_X -v $s:$V/$f:ro"
done
shim/run.sh pytest-newtests-on-base $R -- flock $L docker run --rm --gpus all --ipc=host --name r3-956-pyt-x $ENVS --entrypoint "" -w $V $M_X $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs tests/models/test_deepseek_v4_1_compressor.py::test_sp_ratio2_projection_matches_full_rows tests/models/test_deepseek_v4_1_compressor_prepare.py::test_sp_ratio2_compressor_projects_local_rows_and_gathers_values_and_gates tests/models/test_deepseek_v4_1_ced_model.py::test_ced_encoder_only_plan_stops_at_boundary_after_global_kv_prepare
echo BATCH_DONE
