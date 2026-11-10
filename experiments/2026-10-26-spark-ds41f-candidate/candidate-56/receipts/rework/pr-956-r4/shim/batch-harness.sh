#!/usr/bin/env bash
# Attention harness on sp8 (r4, compressor tp1 mode): base (candidate-54) and cand (+ r4 956)
# at TP4 and TP3 rank shapes, serving environment, two boots each (boot 2 in the reverse arm
# order); then one run per arm with breakable graphs. Each run also does the engine-grid
# decode check (without SP). Every GPU command under flock $L.
. $HOME/spark-lab/agents/round3-956-r4/shim/env.sh
cd $B
for boot in 1 2; do
  if [ $boot = 1 ]; then arms="base cand"; else arms="cand base"; fi
  for world in 4 3; do
    for arm in $arms; do
      case $arm in base) M=$M_BASE;; cand) M=$M_CAND;; esac
      shim/run.sh harness-$arm-tp$world-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-h-$arm-$world-$boot $ENVS --entrypoint "" -w $V $M $HUB $SH $TL $R python3 /shim/attn_harness.py --world $world --arm $arm --compressor tp1 --out /out/$arm-tp$world-boot$boot
    done
  done
done
for arm in base cand; do
  case $arm in base) M=$M_BASE;; cand) M=$M_CAND;; esac
  shim/run.sh harness-$arm-tp4-breakable $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-h-$arm-4-brk $ENVS_BREAKABLE --entrypoint "" -w $V $M $HUB $SH $TL $R python3 /shim/attn_harness.py --world 4 --arm $arm --compressor tp1 --out /out/$arm-tp4-breakable
done
echo BATCH_DONE
