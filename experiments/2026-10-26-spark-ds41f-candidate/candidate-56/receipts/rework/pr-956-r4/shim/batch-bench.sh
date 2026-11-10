#!/usr/bin/env bash
# #956's own mark (r4) on sp8 (one GPU, TP4 rank shapes, compressor rank mode): per-layer SP
# prefill step time of layers 2 (ratio 2) and 20 (ratio 1) at 4K..64K-row chunks, and each
# arm's static memory at kernel declaration. Same session, ABBA: base, cand, cand, base.
. $HOME/spark-lab/agents/round3-956-r4/shim/env.sh
cd $B
for arm in base cand cand base; do
  case $arm in base) M=$M_BASE;; cand) M=$M_CAND;; esac
  n=$(ls out/json/bench-$arm-boot*.json 2>/dev/null | wc -l); boot=$((n+1))
  shim/run.sh bench-$arm-boot$boot $R -- flock $L docker run --rm --gpus all --ipc=host --name r4-956-bench-$arm-$boot $ENVS --entrypoint "" -w $V $M $HUB $SH $TL $R python3 /shim/bench_layers.py --world 4 --arm $arm --out /out/bench-$arm-boot$boot
done
echo BATCH_DONE
