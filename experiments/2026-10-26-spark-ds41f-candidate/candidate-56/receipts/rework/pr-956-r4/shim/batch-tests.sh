#!/usr/bin/env bash
# Unit tests on sp8 (r4), every GPU command under flock $L, a fresh container each: the
# rebase's new tests first (quick), then the full file list on cand x 2 boots and base x 1,
# then the rebase's three test files over the base's code (its new tests must fail there).
. $HOME/spark-lab/agents/round3-956-r4/shim/env.sh
cd $B
NEW="tests/models/test_deepseek_v4_1_compressor.py::test_sp_ratio2_projection_matches_full_rows tests/models/test_deepseek_v4_1_compressor_prepare.py::test_sp_ratio2_compressor_projects_local_rows_and_gathers_values_and_gates tests/models/test_deepseek_v4_1_ced_model.py::test_ced_encoder_only_plan_stops_at_boundary_after_global_kv_prepare"
FILES="tests/models/test_deepseek_v4_1_compressor_prepare.py tests/models/test_deepseek_v4_1_compressor.py tests/models/test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_ced_model.py tests/models/test_deepseek_v4_1_ced_sp.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/test_deepseek_v41_tilelang_front.py tests/models/test_deepseek_v4_1_prepare.py tests/v1/attention/test_b12x_v41_ced_metadata.py tests/v1/worker/test_gpu_model_runner_v2.py tests/kernels/attention/test_deepseek_v41_ced_replay.py"
shim/run.sh pytest-newtests-cand $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-pyt-new $ENVS --entrypoint "" -w $V $M_CAND $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs $NEW
for boot in 1 2; do
  shim/run.sh pytest-cand-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-pyt-cand-$boot $ENVS --entrypoint "" -w $V $M_CAND $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs $FILES
done
shim/run.sh pytest-base-boot1 $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-pyt-base-1 $ENVS --entrypoint "" -w $V $M_BASE $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs $FILES
M_X=""
for f in $(git -C $B/tree-base diff --name-only $IMG_BASE HEAD); do
  s=$B/tree-base/$f
  case $f in tests/models/test_deepseek_v4_1_ced_model.py|tests/models/test_deepseek_v4_1_compressor_prepare.py|tests/models/test_deepseek_v4_1_compressor.py) s=$B/tree-cand/$f;; esac
  M_X="$M_X -v $s:$V/$f:ro"
done
shim/run.sh pytest-newtests-on-base $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-pyt-x $ENVS --entrypoint "" -w $V $M_X $TL $R python3 -m pytest --noconftest -p no:cacheprovider -q -rfEs $NEW
echo BATCH_DONE
