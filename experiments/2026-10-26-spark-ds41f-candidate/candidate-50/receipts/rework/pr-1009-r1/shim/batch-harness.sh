#!/usr/bin/env bash
# Attention harness on sp8: candidate-50 as built (c50), with the corrected 0778 (fix)
# and without 0778 (no1009), at TP3, TP4 and TP8 rank shapes, two boots each (a fresh
# container per run), boot 2 in the reverse arm order. Every GPU command under flock $L.
. $HOME/spark-lab/agents/candidate-50-rework-1009/shim/env.sh
cd $B
TL="-v $B/tlcache:/root/.tilelang/cache"
for boot in 1 2; do
  if [ $boot = 1 ]; then arms="c50 fix no1009"; else arms="no1009 fix c50"; fi
  for world in 4 3 8; do
    for arm in $arms; do
      case $arm in c50) M=$M_C50;; fix) M=$M_FIX;; no1009) M=$M_NO;; esac
      shim/run.sh harness-$arm-tp$world-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name c50rw1009-h-$arm-$world-$boot --entrypoint "" -w $V $M $HUB $SH $TL $R python3 /shim/attn_harness.py --world $world --arm $arm --out /out/$arm-tp$world-boot$boot
    done
  done
done
echo BATCH_DONE
