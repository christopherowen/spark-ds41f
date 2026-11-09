#!/usr/bin/env bash
# Own mark and tests on sp8, every GPU command under flock $L, a fresh container each.
# A = candidate-50 with the corrected 0778 (fix), B = candidate-50 without 0778 (no1009).
# Boot 1 runs A then B, boot 2 B then A (ABBA).
. $HOME/spark-lab/agents/candidate-50-rework-1009/shim/env.sh
cd $B
TL="-v $B/tlcache:/root/.tilelang/cache"
PRB="-v $B/shim/bench_sp_attention_front.py:/tmp/bench_sp_attention_front.py:ro"
own() {  # PR #1009's own bench, unchanged (defaults: 40 warmup, 200 timed iterations)
  shim/run.sh bench-own-fix-boot$1 $R -- flock $L docker run --rm --gpus all --ipc=host --name c50rw1009-own-$1 --entrypoint "" -w $V $M_FIX $PRB $SH $TL $R python3 /tmp/bench_sp_attention_front.py --json-out /out/bench-own-fix-boot$1.json
}
abba() {  # <arm> <mounts> <boot>
  shim/run.sh bench-abba-$1-boot$3 $R -- flock $L docker run --rm --gpus all --ipc=host --name c50rw1009-abba-$1-$3 --entrypoint "" -w $V $2 $SH $TL $R python3 /shim/bench_abba.py /out/bench-abba-$1-boot$3.json
}
# The attention harness once more on the final corrected tree (the shipped docstring).
for world in 4 3 8; do
  shim/run.sh harness-fix-tp$world-final $R -- flock $L docker run --rm --gpus all --ipc=host --name c50rw1009-h-fix-$world-final --entrypoint "" -w $V $M_FIX $HUB $SH $TL $R python3 /shim/attn_harness.py --world $world --arm fix --out /out/fix-tp$world-boot3
done
own 1; abba fix "$M_FIX" 1; abba no1009 "$M_NO" 1
abba no1009 "$M_NO" 2; abba fix "$M_FIX" 2; own 2

# Tests. The file list #1009's own-mark receipts ran, plus the TileLang attention kernels.
FILES="tests/kernels/test_deepseek_v41_tilelang_front.py tests/kernels/test_deepseek_v41_tilelang_columns.py tests/models/test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/models/test_deepseek_v4_1_prepare.py tests/models/test_deepseek_v41_serving_contract.py tests/kernels/attention/test_deepseek_v41_tilelang.py"
for boot in 1 2; do
  shim/run.sh pytest-fix-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c50rw1009-pyt-$boot --entrypoint "" -w $V $M_FIX $TL $R pytest --noconftest -p no:cacheprovider -rf $FILES
done
# The corrected test file over candidate-50's attention.py (original 0778): its decode
# cases must fail there.
T=tests/models/test_deepseek_v4_1_attention.py
M_X=""; for f in $(git -C $B/vllm-c50 diff --name-only base1107 HEAD); do s=$B/vllm-c50/$f; [ "$f" = "$T" ] && s=$B/vllm-fix/$f; M_X="$M_X -v $s:$V/$f:ro"; done
shim/run.sh pytest-newtest-on-c50 $R -- flock $L docker run --rm --gpus all --ipc=host --name c50rw1009-pyt-c50 --entrypoint "" -w $V $M_X $TL $R pytest --noconftest -p no:cacheprovider -rf $T -k test_sp_prefill_with_replicated_fused_wqa_wkv_gathers_mxfp8_and_swa_records
echo BATCH_DONE
