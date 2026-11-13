#!/bin/bash
# After final.sh: #1001-r1 screen A/B with slot-balanced layers (sab) on the c51r1 arm, two
# boots, then the screen diagnosis (base vs #963 decode on the certified screen), two boots.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; RW=$W/bin/rw.sh
while kill -0 2839360 2>/dev/null; do sleep 20; done
for k in 1 2; do $RW diag d963-sab-boot$k c51r1 -- python3 /h/d963.py sab /out/json/d963-sab-boot$k.json; done
for k in 1 2; do $RW diag d963-screen-boot$k c51 -- python3 /h/d963.py screen /out/json/d963-screen-boot$k.json; done
